import argparse
import copy
import os
import pickle
import random


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_infos(data):
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    return data


def set_infos(data, infos):
    if isinstance(data, dict) and "infos" in data:
        data["infos"] = infos
        return data
    return infos


def make_cam_only(data):
    data = copy.deepcopy(data)
    infos = get_infos(data)

    for info in infos:
        if "radars" in info:
            del info["radars"]

    return set_infos(data, infos)


def make_cam_radar(data):
    return copy.deepcopy(data)


def make_radar_front_only(data):
    data = copy.deepcopy(data)
    infos = get_infos(data)

    for info in infos:
        radars = info.get("radars", {})
        if "RADAR_FRONT" in radars:
            info["radars"] = {"RADAR_FRONT": radars["RADAR_FRONT"]}
        else:
            info["radars"] = {}

    return set_infos(data, infos)


def make_radar_drop_all(data):
    data = copy.deepcopy(data)
    infos = get_infos(data)

    for info in infos:
        info["radars"] = {}

    return set_infos(data, infos)


def make_radar_shuffle(data, seed=42):
    data = copy.deepcopy(data)
    infos = get_infos(data)

    radar_list = [copy.deepcopy(info.get("radars", {})) for info in infos]

    rng = random.Random(seed)
    rng.shuffle(radar_list)

    for info, shuffled_radars in zip(infos, radar_list):
        info["radars"] = shuffled_radars

    return set_infos(data, infos)


def process_one_split(input_pkl, output_dir, split_name):
    data = load_pickle(input_pkl)

    variants = {
        "cam_only": make_cam_only(data),
        "cam_radar": make_cam_radar(data),
        "radar_front_only": make_radar_front_only(data),
        "radar_drop_all": make_radar_drop_all(data),
        "radar_shuffle": make_radar_shuffle(data, seed=42),
    }

    for name, variant in variants.items():
        out_path = os.path.join(
            output_dir,
            f"nuscenes-mini_infos_{split_name}_{name}.pkl",
        )
        dump_pickle(variant, out_path)
        print(f"[OK] saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-pkl",
        default="data/nuscenes_anno_pkls/nuscenes-mini_infos_train_radar.pkl",
    )
    parser.add_argument(
        "--val-pkl",
        default="data/nuscenes_anno_pkls/nuscenes-mini_infos_val_radar.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/nuscenes_anno_pkls/ablation",
    )
    args = parser.parse_args()

    process_one_split(args.train_pkl, args.output_dir, "train")
    process_one_split(args.val_pkl, args.output_dir, "val")


if __name__ == "__main__":
    main()
