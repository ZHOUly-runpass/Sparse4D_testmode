import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_projects.datasets.pipelines.loading_radar import LoadRadarPointsFromInfo


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "infos" in data:
        return data["infos"]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        default="data/nuscenes_anno_pkls/nuscenes-mini_infos_train_radar.pkl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of samples to check; 0 checks every sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    infos = load_infos(args.pkl)

    loader = LoadRadarPointsFromInfo(
        max_points=2000,
        point_cloud_range=(-60, -60, -5, 60, 60, 5),
        data_root="data/nuscenes",
        target_frame="lidar",
        strict=True,
        random_seed=args.seed,
    )

    print(f"[INFO] loaded infos: {len(infos)}")

    valid_count = 0
    total_points = 0
    channel_ids = set()
    num_checked = len(infos) if args.limit <= 0 else min(args.limit, len(infos))

    for idx in range(num_checked):
        info = infos[idx]

        results = {
            "radars": info["radars"],
            "lidar2ego_translation": info["lidar2ego_translation"],
            "lidar2ego_rotation": info["lidar2ego_rotation"],
        }

        results = loader(results)

        radar_points = results["radar_points"]

        if radar_points.dtype != np.float32:
            raise AssertionError(f"sample {idx}: expected float32")
        if radar_points.ndim != 2 or radar_points.shape[1] != 7:
            raise AssertionError(f"sample {idx}: invalid shape {radar_points.shape}")
        if not np.isfinite(radar_points).all():
            raise AssertionError(f"sample {idx}: NaN or Inf found")

        if radar_points.shape[0] > 0:
            valid_count += 1
            total_points += radar_points.shape[0]
            channel_ids.update(radar_points[:, 6].astype(np.int64).tolist())

        if args.verbose:
            print("=" * 80)
            print("sample index:", idx)
            print("token:", info["token"])
            print("radar_valid:", results["radar_valid"])
            print("radar_count:", results["radar_count"])
            print("radar_points shape:", radar_points.shape)

    print("=" * 80)
    print("[SUMMARY]")
    print("checked samples:", num_checked)
    print("valid radar samples:", valid_count)
    print("average radar points:", total_points / max(num_checked, 1))
    print("channel ids:", sorted(channel_ids))

    if valid_count != num_checked:
        raise AssertionError(
            f"expected every sample to contain radar points: "
            f"{valid_count}/{num_checked}"
        )
    if channel_ids != {0, 1, 2, 3, 4}:
        raise AssertionError(f"expected five radar channels, got {channel_ids}")

    print("[OK] radar loader validation passed")


if __name__ == "__main__":
    main()
