import argparse
import os
import pickle
from typing import Dict

from nuscenes.nuscenes import NuScenes


RADAR_CHANNELS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_radar_dict(nusc: NuScenes, sample: Dict, dataroot: str) -> Dict:
    radars = {}

    for channel in RADAR_CHANNELS:
        if channel not in sample["data"]:
            continue

        sd_token = sample["data"][channel]
        sd_rec = nusc.get("sample_data", sd_token)
        cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])

        radars[channel] = {
            "sample_data_token": sd_token,
            "data_path": os.path.join(dataroot, sd_rec["filename"]),
            "filename": sd_rec["filename"],
            "timestamp": sd_rec["timestamp"],
            "sensor2ego_translation": cs_rec["translation"],
            "sensor2ego_rotation": cs_rec["rotation"],
            "ego2global_translation": pose_rec["translation"],
            "ego2global_rotation": pose_rec["rotation"],
            "is_key_frame": sd_rec["is_key_frame"],
        }

    return radars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    args = parser.parse_args()

    print(f"[INFO] loading NuScenes: {args.dataroot}, {args.version}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    print(f"[INFO] loading pkl: {args.input_pkl}")
    data = load_pickle(args.input_pkl)

    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
        pkl_type = "dict_infos"
    elif isinstance(data, list):
        infos = data
        pkl_type = "list"
    else:
        raise TypeError(f"Unsupported pkl type: {type(data)}")

    token_to_sample = {s["token"]: s for s in nusc.sample}

    missing = 0
    for info in infos:
        token = info.get("token", None)
        if token is None or token not in token_to_sample:
            missing += 1
            info["radars"] = {}
            continue

        sample = token_to_sample[token]
        info["radars"] = build_radar_dict(nusc, sample, args.dataroot)

    print(f"[INFO] augmented infos: {len(infos)}, missing token: {missing}")

    if pkl_type == "dict_infos":
        data["infos"] = infos
        output = data
    else:
        output = infos

    dump_pickle(output, args.output_pkl)
    print(f"[OK] saved: {args.output_pkl}")


if __name__ == "__main__":
    main()
