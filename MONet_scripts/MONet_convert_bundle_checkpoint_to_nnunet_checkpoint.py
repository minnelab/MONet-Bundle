#!/usr/bin/env python

import os
import json
import yaml
import torch

try:
    from monai.apps.nnunet.nnunet_bundle import convert_monai_bundle_to_nnunet
except ImportError:
    convert_monai_bundle_to_nnunet = None


def convert(dataset_name_or_id: str, bundle_root: str, fold: int):
    if os.path.exists(os.path.join(bundle_root, "configs", "inference.json")):
        with open(os.path.join(bundle_root, "configs", "inference.json"), "r") as f:
            inference_config = json.load(f)
    elif os.path.exists(os.path.join(bundle_root, "configs", "inference.yaml")):
        with open(os.path.join(bundle_root, "configs", "inference.yaml"), "r") as f:
            inference_config = yaml.safe_load(f)
    else:
        raise ValueError("No inference configuration found in the bundle")
    nnunet_config = {"dataset_name_or_id": dataset_name_or_id}
    nnunet_config["nnunet_trainer"] = inference_config["nnunet_trainer_class_name"]
    nnunet_config["nnunet_plans"] = "nnUNetPlans"
    nnunet_config["nnunet_config"] = inference_config["nnunet_configuration"]
    inference_config["nnunet_config_ckpt"]["init_args"] = {"configuration": inference_config["nnunet_configuration"]}
    inference_config["nnunet_config_ckpt"]["trainer_name"] = inference_config["nnunet_trainer_class_name"]

    torch.save(inference_config["nnunet_config_ckpt"], os.path.join(bundle_root, "models", "nnunet_checkpoint.pth"))
    with open(os.path.join(bundle_root, "models", "dataset.json"), "w") as f:
        json.dump(inference_config["dataset_json"], f)
    with open(os.path.join(bundle_root, "models", "plans.json"), "w") as f:
        json.dump(inference_config["plans"], f)
    convert_monai_bundle_to_nnunet(nnunet_config=nnunet_config, bundle_root_folder=bundle_root, fold=fold)


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert MONAI Bundle checkpoint to nnUNet checkpoint")
    parser.add_argument("--dataset_name_or_id", type=str, required=True, help="Dataset name or ID to convert")
    parser.add_argument("--bundle_root", type=str, required=True, help="Root folder for the MONAI Bundle")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for the dataset")
    return parser


def main():
    args = get_arg_parser().parse_args()
    dataset_name_or_id = args.dataset_name_or_id
    bundle_root = args.bundle_root
    fold = args.fold
    convert(dataset_name_or_id, bundle_root, fold)


if __name__ == "__main__":
    main()
