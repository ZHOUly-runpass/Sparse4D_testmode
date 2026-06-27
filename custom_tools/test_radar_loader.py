import pickle
import numpy as np

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

    print(f"[INFO] loaded infos: {len(infos)}")

    valid_count = 0

    for idx in range(min(10, len(infos))):
        info = infos[idx]

        results = {
            "radars": info["radars"],
        }

        results = loader(results)

        radar_points = results["radar_points"]

        print("=" * 80)
        print("sample index:", idx)
        print("token:", info["token"])
        print("radar_valid:", results["radar_valid"])
        print("radar_count:", results["radar_count"])
        print("radar_points shape:", radar_points.shape)

        if radar_points.shape[0] > 0:
            valid_count += 1
            print("first point:", radar_points[0])
            print("xyz min:", radar_points[:, 0:3].min(axis=0))
            print("xyz max:", radar_points[:, 0:3].max(axis=0))
            print("velocity mean:", radar_points[:, 3:5].mean(axis=0))
            print("rcs mean:", np.mean(radar_points[:, 5]))
            print("channel ids:", np.unique(radar_points[:, 6]))

    print("=" * 80)
    print("[SUMMARY]")
    print("checked samples:", min(10, len(infos)))
    print("valid radar samples:", valid_count)


if __name__ == "__main__":
    main()
