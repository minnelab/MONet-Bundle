#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import requests
import SimpleITK as sitk

from MONet.utils import get_available_models

try:
    import torch
    from monai.bundle import ConfigParser
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SaveImage
except ImportError:
    torch = None
    ConfigParser = None
    LoadImaged = None
    EnsureChannelFirstd = None
    Compose = None
    SaveImage = None


def get_arg_parser():

    parser = argparse.ArgumentParser(description="Run Local segmentation inference using the MAIA Segmentation Portal.")

    parser.add_argument("--input-image", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-folder", "-o", type=str, required=True, help="Folder to save the output predictions")
    parser.add_argument("--username", required=True, help="Username for MAIA Segmentation Portal")
    if False:
        ...
        # temp_args, _ = parser.parse_known_args()
        # home = os.path.expanduser("~")
        # auth_path = os.path.join(home, ".monet", f"{temp_args.username}_auth.json")
        # with open(auth_path, "r") as token_file:
        # token_data = json.load(token_file)
        # token = token_data.get("access_token")
        # models = get_available_models(token, temp_args.username)
    parser.add_argument("--model", required=True, help="Model to use for segmentation")
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    model_name = args.model

    home = os.path.expanduser("~")
    model_path = os.path.join(home, ".monet", "models", model_name + ".ts")

    if not os.path.exists(model_path):
        auth_path = os.path.join(home, ".monet", f"{args.username}_auth.json")
        with open(auth_path, "r") as token_file:
            token_data = json.load(token_file)
            token = token_data.get("access_token")
            models = get_available_models(token, args.username)
            maia_segmentation_portal_url = models.get(model_name)
            if not maia_segmentation_portal_url:
                raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(models.keys())}")

            model_url = f"{maia_segmentation_portal_url}model/MONetBundle"
        if not token:
            raise ValueError("Access token not found in the token file.")

        headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}  # Replace with your actual token
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url, headers=headers)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(response.content)


    extra_files = {"inference.json": "", "metadata.json": ""}
    model = torch.jit.load(model_path, _extra_files=extra_files)

    inference = json.loads(extra_files["inference.json"])
    metadata = json.loads(extra_files["metadata.json"])
    
    model_metadata = metadata['network_data_format']
    required_input_channels = model_metadata["inputs"]
    print(f"Required input channels: {len(required_input_channels)}")
    for idx, channel in enumerate(required_input_channels):
        print(f"Input Channel {idx}: {channel}")
    print(f"Expected Output Labels: {model_metadata['outputs']['pred']['channel_def']}")
    if args.input_image.endswith(".nii.gz"):
        print(f"Verifying input file: {args.input_image}")
        input_img = sitk.ReadImage(args.input_image)
        print(f"Input image size: {input_img.GetSize()}")
        n_channels = 1
        if len(input_img.GetSize()) == 4:
            n_channels = input_img.GetSize()[3]
        print(f"Number of channels in input image: {n_channels}")
        if n_channels != len(required_input_channels):
            print(f"Error: Expected {len(required_input_channels)} channels, but found {n_channels} in the input image.")
            print("\n")
            print("To create a valid input, use the following command:")
            print(
                "\t MONet_concatenate_modalities --input_folder <input_path> --output_folder <output_path> --ref_modality <reference_modality> --modality-mapping <mapping>"
            )
            print("\n")
            print(
                "\t<input_path> should be a folder containing the input images, and <output_path> will contain the concatenated image."
            )
            print(
                "\t<mapping> should be a comma-separated list of modalities to concatenate, e.g., 'CT:_ct.nii.gz,PT:_pet.nii.gz'."
            )
            print("\t<reference_modality> is the modality to which all other modalities will be resampled, e.g., 'CT'.")
            return
        else:
            print(f"Input file {args.input_image} is valid for the model {model_name}.")

    print(json.dumps(metadata, indent=4))
    parser = ConfigParser(inference)

    nnunet_predictor = parser.get_parsed_content("network_def", instantiate=True)
    nnunet_predictor.predictor.network = model

    # Define the transforms
    transforms = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])

    # Load and transform the input image
    data = transforms({"image": args.input_image})

    # Perform prediction

    pred = nnunet_predictor(data["image"][None])

    # Save the prediction
    SaveImage(output_dir=args.output_folder, separate_folder=False, output_postfix="segmentation", output_ext=".nii.gz")(pred[0])


if __name__ == "__main__":
    main()
