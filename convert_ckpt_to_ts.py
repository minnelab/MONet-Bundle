#!/usr/bin/env python

from monai.bundle.scripts import ckpt_export
from pathlib import Path
import yaml
import json

def export(bundle_root, checkpoint_name, nnunet_trainer_name="nnUNetTrainer", fold=0):
    
    with open(Path(bundle_root).joinpath("models","plans.json"),"r") as f:
        plans = json.load(f)

    with open(Path(bundle_root).joinpath("configs","plans.yaml"),"w") as f:
        yaml.dump({"plans": plans}, f)

    with open(Path(bundle_root).joinpath("models","dataset.json"),"r") as f:
        dataset_json = json.load(f)
        
    with open(Path(bundle_root).joinpath("configs","dataset.yaml"),"w") as f:
        yaml.dump({"dataset_json": dataset_json}, f)
        
    with open(str(Path(bundle_root).joinpath("configs","inference.yaml")), "r") as f:
        config = yaml.safe_load(f)
    
    with open(str(Path(bundle_root).joinpath("configs","dataset.yaml")), "r") as f:
        dataset_config = yaml.safe_load(f)
        config["dataset_json"] = dataset_config["dataset_json"]
    with open(str(Path(bundle_root).joinpath("configs","plans.yaml")), "r") as f:
        plans = yaml.safe_load(f)
        config["plans"] = plans["plans"]
    
    config["nnunet_trainer_class_name"] = nnunet_trainer_name

    config["network_def_predictor"] = "$@network_def.network_weights"
    
    with open(str(Path(bundle_root).joinpath("configs","inference.yaml")), "w") as f:
        yaml.dump(config, f)
    
    ckpt_export(
    net_id = "network_def_predictor",
    key_in_ckpt="network_weights",
    filepath = str(Path(bundle_root).joinpath("models", f"fold_{fold}", "model.ts")),
    ckpt_file = str(Path(bundle_root).joinpath("models", f"fold_{fold}", checkpoint_name )),
    config_file = str(Path(bundle_root).joinpath("configs", "inference.yaml")),
    bundle_root = bundle_root,
    model_name = checkpoint_name,
    #input_shape = [1, 1, 96, 96, 96],
    #use_trace = True
    )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Convert nnUNet checkpoint to TorchScript")
    parser.add_argument("--bundle_root", type=str, required=True, help="Path to the nnUNet bundle root")
    parser.add_argument("--checkpoint_name", type=str, default="model.pt", help="Checkpoint name")
    parser.add_argument("--nnunet_trainer_name", type=str, default="nnUNetTrainer", help="Trainer name")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument(
        "--dataset_name_or_id",
        type=str,
        required=False,
        default="00",
        help="Dataset name or ID to convert",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle_root = args.bundle_root
    checkpoint_name = args.checkpoint_name
    nnunet_trainer_name = args.nnunet_trainer_name
    fold = args.fold
    export(bundle_root=bundle_root, checkpoint_name=checkpoint_name, nnunet_trainer_name=nnunet_trainer_name, fold=fold)