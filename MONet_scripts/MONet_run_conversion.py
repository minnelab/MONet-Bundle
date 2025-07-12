#!/usr/bin/env python
from __future__ import annotations

try:
    from python_on_whales import docker
except ImportError:
    docker = None


import os
import zipfile


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert nnUNet model to MONAI Bundle format.")
    parser.add_argument("--bundle_path", type=str, required=True, help="Path of the MONet Bundle to save the converted model.")
    parser.add_argument(
        "--nnunet_model", type=str, required=True, help="Path to the nnUNet model checkpoint file (e.g., Task09_Spleen.zip)."
    )
    parser.add_argument("--dataset_name_or_id", type=str, required=True, help="Name or ID of the dataset to convert.")

    return parser


def main():

    parser = get_arg_parser()

    args = parser.parse_args()

    bundle_path = args.bundle_path

    os.makedirs("nnUNet_Models", exist_ok=True)

    with zipfile.ZipFile(args.nnunet_model, "r") as zip_ref:
        zip_ref.extractall("nnUNet_Models")

    docker.run(
        "nnunet-monai-bundle-converter",
        gpus="device=0",
        envs={"nnUNet_results": "/input"},
        command=["--fold", "0", "--bundle_root", "/model/bundle", "--dataset_name_or_id", args.dataset_name_or_id],
        volumes=[(bundle_path, "/model/bundle"), ("./nnUNet_Models", "/input")],
        shm_size="2g",
    )

    model_folder = os.path.join(bundle_path, "models", "fold_0")

    print("Conversion completed successfully.")
    print(f"Please check the folder {bundle_path} for the converted nnUNet model in the MONAI Bundle format.")
    print(f"You can find the TorchScript model in the {model_folder} folder.")


if __name__ == "__main__":
    main()
