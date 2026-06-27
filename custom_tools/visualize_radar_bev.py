import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
    pkl_path = "data/nuscenes_anno_pkls/nuscenes-mini_infos_train_radar.pkl"
    infos = load_infos(pkl_path)

    loader = LoadRadarPointsFromInfo(
        max_points=2000,
        point_cloud_range=(-60, -60, -5, 60, 60, 5),
    )

    os.makedirs("outputs", exist_ok=True)

    for idx in range(3):
        info = infos[idx]
        results = loader(
            {
                "radars": info["radars"],
                "lidar2ego_translation": info["lidar2ego_translation"],
                "lidar2ego_rotation": info["lidar2ego_rotation"],
            }
        )
        points = results["radar_points"]

        plt.figure(figsize=(7, 7))

        if points.shape[0] > 0:
            plt.scatter(points[:, 0], points[:, 1], s=4)

        plt.xlim(-60, 60)
        plt.ylim(-60, 60)
        plt.xlabel("x in lidar frame")
        plt.ylabel("y in lidar frame")
        plt.title(f"Radar BEV sample {idx}, count={points.shape[0]}")
        plt.grid(True)

        save_path = f"outputs/radar_bev_{idx}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"[OK] saved {save_path}")


if __name__ == "__main__":
    main()
