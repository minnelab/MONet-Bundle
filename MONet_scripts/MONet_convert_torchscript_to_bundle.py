#!/usr/bin/env python

from MONet_fetch_bundle import fetch_bundle
import shutil
import os
import zipfile
import json
import torch
try:
    from monai.apps.nnunet.nnunet_bundle import convert_nnunet_to_monai_bundle
except ImportError:
    convert_nnunet_to_monai_bundle = None


def convert_torchscript_to_pt(torchscript_path: str, pt_path: str):
    model = torch.jit.load(torchscript_path, map_location=torch.device('cpu'))
    state_dict = {"network_weights": model.state_dict()}
    torch.save(state_dict, pt_path)


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert TorchScript to MONAI Bundle format")
    parser.add_argument("--torchscript_path", type=str, required=True, help="Path to the TorchScript model")
    parser.add_argument("--bundle_root", type=str, required=True, help="Location to save the MONAI Bundle")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for the model")
    return parser


def main():
    args = get_arg_parser().parse_args()
    torchscript_path = args.torchscript_path
    bundle_root = args.bundle_root
    fold = args.fold
    fetch_bundle(bundle_root)
    
    os.remove(os.path.join(bundle_root,"MONetBundle", "configs", "metadata.json"))
    os.remove(os.path.join(bundle_root,"MONetBundle", "configs", "inference.yaml"))
    os.makedirs(os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}"), exist_ok=True)
    shutil.copy(torchscript_path, os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.ts"))


    with zipfile.ZipFile(os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.ts"), "r") as zf:
        # Load the "inference.json" file stored in the 'extra/' dir
        raw_bytes = zf.read("model/extra/metadata.json")  # still bytes
        json_str = raw_bytes.decode("utf-8")         # convert to string
        data = json.loads(json_str)                  # parse JSON

    with open(os.path.join(bundle_root,"MONetBundle", "configs", "metadata.json"),"w") as f:
        json.dump(data,f)

    with zipfile.ZipFile(os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.ts"), "r") as zf:
        # Load the "inference.json" file stored in the 'extra/' dir
        raw_bytes = zf.read("model/extra/inference.json")  # still bytes
        json_str = raw_bytes.decode("utf-8")         # convert to string
        data = json.loads(json_str)                  # parse JSON
        
    with open(os.path.join(bundle_root,"MONetBundle", "configs", "inference.json"),"w") as f:
        json.dump(data,f)
        
    # Convert TorchScript chekpoint to PT checkpoint
    convert_torchscript_to_pt(os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.ts"), os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.pt"))
    convert_torchscript_to_pt(os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.ts"), os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "checkpoint_epoch=1000.pt"))
    convert_torchscript_to_pt(os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "model.ts"), os.path.join(bundle_root,"MONetBundle", "models", f"fold_{fold}", "checkpoint_key_metric=1.0.pt"))
if __name__ == "__main__":
    main()
