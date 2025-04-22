#!/usr/bin/env python

from monai.apps.nnunet.nnunet_bundle import convert_nnunet_to_monai_bundle


def main(dataset_name_or_id: str, bundle_root: str, fold: int):    
    nnunet_config = {
        "dataset_name_or_id": dataset_name_or_id,
    }

    convert_nnunet_to_monai_bundle(
        nnunet_config=nnunet_config,
        bundle_root_folder=bundle_root,
        fold=fold,
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert nnUNet checkpoint to MONAI Bundle checkpoint")
    parser.add_argument(
        "--dataset_name_or_id",
        type=str,
        required=True,
        help="Dataset name or ID to convert",
    )
    parser.add_argument(
        "--bundle_root",
        type=str,
        required=True,
        help="Root folder for the MONAI Bundle",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for the dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_name_or_id = args.dataset_name_or_id
    bundle_root = args.bundle_root
    fold = args.fold
    main(dataset_name_or_id, bundle_root, fold)
