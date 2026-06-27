import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud


def transform_points(points_xyz, translation, rotation):
    if points_xyz.shape[0] == 0:
        return points_xyz

    rot = Quaternion(rotation).rotation_matrix
    trans = np.array(translation, dtype=np.float32).reshape(1, 3)

    return points_xyz @ rot.T + trans


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

    def _safe_column(self, points, index, default=0.0):
        if points.shape[1] > index:
            return points[:, index:index + 1].astype(np.float32)
        return np.full((points.shape[0], 1), default, dtype=np.float32)

    def _load_one_radar(self, radar_info, channel_id):
        path = radar_info["data_path"]

        if not os.path.exists(path):
            print(f"[WARN] radar file not found: {path}")
            return np.zeros((0, 7), dtype=np.float32)

        pc = RadarPointCloud.from_file(path)
        points = pc.points.T

        if points.shape[0] == 0:
            return np.zeros((0, 7), dtype=np.float32)

        xyz = points[:, 0:3].astype(np.float32)

        xyz_ego = transform_points(
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

        cid = np.full((points.shape[0], 1), channel_id, dtype=np.float32)

        radar_points = np.concatenate(
            [xyz_ego, vx_comp, vy_comp, rcs, cid],
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

        for idx, sensor in enumerate(self.use_sensors):
            if sensor not in radars:
                continue

            pts = self._load_one_radar(radars[sensor], idx)
            all_points.append(pts)

        if len(all_points) == 0:
            radar_points = np.zeros((0, 7), dtype=np.float32)
        else:
            radar_points = np.concatenate(all_points, axis=0)
            radar_points = self._filter_range(radar_points)

        if radar_points.shape[0] > self.max_points:
            choice = np.random.choice(
                radar_points.shape[0],
                self.max_points,
                replace=False,
            )
            radar_points = radar_points[choice]

        results["radar_points"] = radar_points.astype(np.float32)
        results["radar_valid"] = radar_points.shape[0] > 0
        results["radar_count"] = radar_points.shape[0]

        return results
