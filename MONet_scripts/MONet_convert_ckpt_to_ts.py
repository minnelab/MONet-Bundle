#!/usr/bin/env python
from __future__ import annotations

try:
    from monai.bundle.scripts import ckpt_export
except ImportError:
    ckpt_export = None


import json
from pathlib import Path

import yaml


def export(bundle_root, checkpoint_name, nnunet_trainer_name="nnUNetTrainer", nnunet_configuration="3d_fullres", fold=0):

    with open(Path(bundle_root).joinpath("models", "plans.json"), "r") as f:
        plans = json.load(f)

    with open(Path(bundle_root).joinpath("configs", "plans.yaml"), "w") as f:
        yaml.dump({"plans": plans}, f)

    import collections

    def ordered_load_json(filename):
        # Use object_pairs_hook to preserve dict order in JSON
        with open(filename, "r") as f:
            return json.load(f, object_pairs_hook=collections.OrderedDict)

    def represent_ordereddict(dumper, data):
        value = []
        for item_key, item_value in data.items():
            node_key = dumper.represent_data(item_key)
            node_value = dumper.represent_data(item_value)
            value.append((node_key, node_value))
        return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

    yaml.add_representer(collections.OrderedDict, represent_ordereddict)

    dataset_json = ordered_load_json(Path(bundle_root).joinpath("models", "dataset.json"))

    with open(Path(bundle_root).joinpath("configs", "dataset.yaml"), "w") as f:
        yaml.dump({"dataset_json": dataset_json}, f, sort_keys=False)
 

    with open(str(Path(bundle_root).joinpath("configs", "inference.yaml")), "r") as f:
        config = yaml.safe_load(f)

    #with open(str(Path(bundle_root).joinpath("configs", "dataset.yaml")), "r") as f:
    #    dataset_config = yaml.safe_load(f)
    config["dataset_json"] = dataset_json
    with open(str(Path(bundle_root).joinpath("configs", "plans.yaml")), "r") as f:
        plans = yaml.safe_load(f)
        config["plans"] = plans["plans"]

    config["nnunet_trainer_class_name"] = nnunet_trainer_name

    config["nnunet_configuration"] = nnunet_configuration

    config["network_def_predictor"] = "$@network_def.network_weights"

    with open(str(Path(bundle_root).joinpath("configs", "inference.yaml")), "w") as f:
        yaml.dump(config, f)

    ckpt_export(
        net_id="network_def_predictor",
        key_in_ckpt="network_weights",
        filepath=str(Path(bundle_root).joinpath("models", f"fold_{fold}", "model.ts")),
        ckpt_file=str(Path(bundle_root).joinpath("models", f"fold_{fold}", checkpoint_name)),
        config_file=str(Path(bundle_root).joinpath("configs", "inference.yaml")),
        bundle_root=bundle_root,
        model_name=checkpoint_name,
        # input_shape = [1, 1, 96, 96, 96],
        # use_trace = True
    )


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert MONet Bundlecheckpoint to TorchScript")
    parser.add_argument("--bundle_root", type=str, required=True, help="Path to the MONet Bundle root")
    parser.add_argument("--checkpoint_name", type=str, default="model.pt", help="Checkpoint name")
    parser.add_argument("--nnunet_trainer_name", type=str, default="nnUNetTrainer", help="Trainer name")
    parser.add_argument("--nnunet_configuration", type=str, default="3d_fullres", help="nnUNet configuration")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    return parser


def main():
    args = get_arg_parser().parse_args()
    bundle_root = args.bundle_root
    checkpoint_name = args.checkpoint_name
    nnunet_trainer_name = args.nnunet_trainer_name
    nnunet_configuration = args.nnunet_configuration
    fold = args.fold
    export(
        bundle_root=bundle_root,
        checkpoint_name=checkpoint_name,
        nnunet_trainer_name=nnunet_trainer_name,
        nnunet_configuration=nnunet_configuration,
        fold=fold,
    )


if __name__ == "__main__":
    main()
