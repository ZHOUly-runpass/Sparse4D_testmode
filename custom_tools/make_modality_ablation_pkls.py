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


def _radar_tokens(info):
    return tuple(
        sorted(
            (sensor, radar_info.get("sample_data_token"))
            for sensor, radar_info in info.get("radars", {}).items()
        )
    )


def validate_variants(source_data, variants):
    source_infos = get_infos(source_data)
    expected_count = len(source_infos)

    for name, variant in variants.items():
        actual_count = len(get_infos(variant))
        if actual_count != expected_count:
            raise AssertionError(
                f"{name}: expected {expected_count} infos, got {actual_count}"
            )

    if not all("radars" not in info for info in get_infos(variants["cam_only"])):
        raise AssertionError("cam_only must remove the radars field")

    front_infos = get_infos(variants["radar_front_only"])
    if not all(set(info.get("radars", {})) <= {"RADAR_FRONT"} for info in front_infos):
        raise AssertionError("radar_front_only contains a non-front sensor")

    drop_infos = get_infos(variants["radar_drop_all"])
    if not all(info.get("radars") == {} for info in drop_infos):
        raise AssertionError("radar_drop_all must contain empty radar mappings")

    cam_radar_infos = get_infos(variants["cam_radar"])
    if [_radar_tokens(info) for info in cam_radar_infos] != [
        _radar_tokens(info) for info in source_infos
    ]:
        raise AssertionError("cam_radar must preserve source radar assignments")

    shuffled_infos = get_infos(variants["radar_shuffle"])
    if sorted(_radar_tokens(info) for info in shuffled_infos) != sorted(
        _radar_tokens(info) for info in source_infos
    ):
        raise AssertionError("radar_shuffle must preserve the radar assignment multiset")

    mismatch_count = sum(
        _radar_tokens(source) != _radar_tokens(shuffled)
        for source, shuffled in zip(source_infos, shuffled_infos)
    )
    if expected_count > 1 and mismatch_count == 0:
        raise AssertionError("radar_shuffle did not change any assignment")

    print(
        f"[OK] validated {expected_count} infos; "
        f"shuffle mismatches: {mismatch_count}"
    )


def process_one_split(input_pkl, output_dir, split_name):
    data = load_pickle(input_pkl)

    variants = {
        "cam_only": make_cam_only(data),
        "cam_radar": make_cam_radar(data),
        "radar_front_only": make_radar_front_only(data),
        "radar_drop_all": make_radar_drop_all(data),
        "radar_shuffle": make_radar_shuffle(data, seed=42),
    }

    validate_variants(data, variants)

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
