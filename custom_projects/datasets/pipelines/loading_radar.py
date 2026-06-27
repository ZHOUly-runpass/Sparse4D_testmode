import os
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud

try:
    from mmdet.datasets.builder import PIPELINES
except ModuleNotFoundError as exc:
    if exc.name != "mmcv._ext":
        raise

    class _LocalOnlyRegistry:
        """No-op registry used only by the local CPU validation environment."""

        @staticmethod
        def register_module():
            return lambda cls: cls

    PIPELINES = _LocalOnlyRegistry()


def transform_points(points_xyz, translation, rotation):
    if points_xyz.shape[0] == 0:
        return points_xyz

    rot = Quaternion(rotation).rotation_matrix
    trans = np.array(translation, dtype=np.float32).reshape(1, 3)

    return points_xyz @ rot.T + trans


def inverse_transform_points(points_xyz, translation, rotation):
    if points_xyz.shape[0] == 0:
        return points_xyz

    rot = Quaternion(rotation).rotation_matrix
    trans = np.array(translation, dtype=np.float32).reshape(1, 3)
    return (points_xyz - trans) @ rot


def transform_vectors(vectors_xyz, rotation):
    if vectors_xyz.shape[0] == 0:
        return vectors_xyz

    rot = Quaternion(rotation).rotation_matrix
    return vectors_xyz @ rot.T


def inverse_transform_vectors(vectors_xyz, rotation):
    if vectors_xyz.shape[0] == 0:
        return vectors_xyz

    rot = Quaternion(rotation).rotation_matrix
    return vectors_xyz @ rot


@PIPELINES.register_module()
class LoadRadarPointsFromInfo:
    """
    Load nuScenes radar points from info["radars"].

    Output:
        results["radar_points"]: [N, 7]

    Field order:
        x, y, z, vx_comp, vy_comp, rcs, channel_id
    """

    def __init__(
        self,
        use_sensors=None,
        max_points=2000,
        point_cloud_range=(-60, -60, -5, 60, 60, 5),
        data_root=None,
        target_frame="lidar",
        strict=False,
        random_seed=None,
        dropout_prob=0.0,
    ):
        self.use_sensors = use_sensors or [
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
            "RADAR_BACK_LEFT",
            "RADAR_BACK_RIGHT",
        ]
        self.max_points = max_points
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.data_root = data_root
        self.target_frame = target_frame
        self.strict = strict
        self.rng = (
            np.random.default_rng(random_seed)
            if random_seed is not None
            else None
        )
        self.dropout_prob = float(dropout_prob)

        if self.target_frame not in {"ego", "lidar"}:
            raise ValueError("target_frame must be 'ego' or 'lidar'")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be in [0, 1]")

    def _resolve_path(self, radar_info):
        candidates = []
        data_path = radar_info.get("data_path")
        filename = radar_info.get("filename")

        if data_path:
            candidates.append(data_path)
        if self.data_root and filename:
            candidates.append(os.path.join(self.data_root, filename))
        if filename:
            candidates.append(filename)

        for path in candidates:
            if os.path.exists(path):
                return path

        missing_path = candidates[0] if candidates else "<missing radar path>"
        if self.strict:
            raise FileNotFoundError(missing_path)
        print(f"[WARN] radar file not found: {missing_path}")
        return None

    def _safe_column(self, points, index, default=0.0):
        if points.shape[1] > index:
            return points[:, index:index + 1].astype(np.float32)
        return np.full((points.shape[0], 1), default, dtype=np.float32)

    def _load_one_radar(self, radar_info, channel_id, results):
        path = self._resolve_path(radar_info)
        if path is None:
            return np.zeros((0, 7), dtype=np.float32)

        pc = RadarPointCloud.from_file(path)
        points = pc.points.T

        if points.shape[0] == 0:
            return np.zeros((0, 7), dtype=np.float32)

        xyz = points[:, 0:3].astype(np.float32)

        xyz_target = transform_points(
            xyz,
            radar_info["sensor2ego_translation"],
            radar_info["sensor2ego_rotation"],
        )

        # nuScenes radar common layout:
        # 0:x, 1:y, 2:z, 3:dyn_prop, 4:id, 5:rcs,
        # 6:vx, 7:vy, 8:vx_comp, 9:vy_comp
        rcs = self._safe_column(points, 5)
        vx_comp = self._safe_column(points, 8)
        vy_comp = self._safe_column(points, 9)
        velocity = np.concatenate(
            [vx_comp, vy_comp, np.zeros_like(vx_comp)], axis=1
        )
        velocity = transform_vectors(
            velocity, radar_info["sensor2ego_rotation"]
        )

        if self.target_frame == "lidar":
            lidar_translation = results.get("lidar2ego_translation")
            lidar_rotation = results.get("lidar2ego_rotation")
            if lidar_translation is None or lidar_rotation is None:
                raise KeyError(
                    "lidar target frame requires lidar2ego_translation "
                    "and lidar2ego_rotation in results"
                )
            xyz_target = inverse_transform_points(
                xyz_target, lidar_translation, lidar_rotation
            )
            velocity = inverse_transform_vectors(velocity, lidar_rotation)

        cid = np.full((points.shape[0], 1), channel_id, dtype=np.float32)

        radar_points = np.concatenate(
            [xyz_target, velocity[:, :2], rcs, cid],
            axis=1,
        )

        return radar_points

    def _filter_range(self, points):
        if points.shape[0] == 0:
            return points

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )

        return points[mask]

    def __call__(self, results):
        radars = results.get("radars", {})
        all_points = []

        if self.dropout_prob > 0.0:
            random_value = (
                self.rng.random() if self.rng is not None else np.random.random()
            )
            if random_value < self.dropout_prob:
                radars = {}

        for idx, sensor in enumerate(self.use_sensors):
            if sensor not in radars:
                continue

            pts = self._load_one_radar(radars[sensor], idx, results)
            all_points.append(pts)

        if len(all_points) == 0:
            radar_points = np.zeros((0, 7), dtype=np.float32)
        else:
            radar_points = np.concatenate(all_points, axis=0)
            radar_points = self._filter_range(radar_points)

        if radar_points.shape[0] > self.max_points:
            sampler = self.rng if self.rng is not None else np.random
            choice = sampler.choice(
                radar_points.shape[0], self.max_points, replace=False
            )
            radar_points = radar_points[choice]

        results["radar_points"] = radar_points.astype(np.float32)
        results["radar_valid"] = radar_points.shape[0] > 0
        results["radar_count"] = radar_points.shape[0]

        return results


@PIPELINES.register_module()
class FormatRadarPoints:
    """Convert variable-length radar points and flags for MMCV collation."""

    def __call__(self, results):
        results["radar_points"] = DC(
            torch.as_tensor(results["radar_points"]).float(), stack=False
        )
        results["radar_valid"] = DC(
            torch.as_tensor(np.array(results["radar_valid"], dtype=np.bool_)),
            stack=True,
        )
        results["radar_count"] = DC(
            torch.as_tensor(np.array(results["radar_count"], dtype=np.int64)),
            stack=True,
        )
        return results
