#!/usr/bin/env python3
import json
import torch
from monai.bundle import ConfigParser
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose, SaveImage
import argparse


def get_arg_parser():

    parser = argparse.ArgumentParser(description="MONet Prediction Script")
    parser.add_argument("--model", type=str, required=True, help="Path to the TorchScript model file")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-folder", type=str, required=True, help="Folder to save the output predictions")
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    extra_files = {"inference.json": "", "metadata.json": ""}
    model = torch.jit.load(args.model,_extra_files=extra_files)

    inference = json.loads(extra_files["inference.json"])
    metadata = json.loads(extra_files["metadata.json"])

    parser = ConfigParser(inference)

    nnunet_predictor = parser.get_parsed_content("network_def",instantiate=True)
    nnunet_predictor.predictor.network = model


    # Define the transforms
    transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"])
    ])

    # Load and transform the input image
    data = transforms({"image": args.input_image})

    # Perform prediction

    pred = nnunet_predictor(data["image"][None])

    # Save the prediction
    SaveImage(output_dir=args.output_folder, separate_folder=False, output_postfix="segmentation", output_ext=".nii.gz")(pred[0])


if __name__ == "__main__":
    main()
