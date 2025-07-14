#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

import requests
import SimpleITK as sitk

from MONet.utils import get_available_models


def run_inference(input_path, output_path, model, username):

    home = os.path.expanduser("~")
    auth_path = os.path.join(home, ".monet", f"{username}_auth.json")
    with open(auth_path, "r") as token_file:
        token_data = json.load(token_file)
        token = token_data.get("access_token")
        models = get_available_models(token, username)
        maia_segmentation_portal_url = models.get(model)
        if not maia_segmentation_portal_url:
            raise ValueError(f"Model '{model}' is not supported. Available models: {list(models.keys())}")
        url = f"{maia_segmentation_portal_url}infer/MONetBundle?output=image"
        if not token:
            raise ValueError("Access token not found in the token file.")

    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}  # Replace with your actual token

    params = {
        # 'device': 'NVIDIA GeForce RTX 2070 SUPER:0',
        # 'model_filename': 'model.ts',
    }

    response = requests.get(f"{maia_segmentation_portal_url}info/", headers=headers)  # Test connection
    if response.status_code != 200:
        raise ConnectionError(f"Failed to connect to MAIA Segmentation Portal: {response.status_code} - {response.text}")
    else:
        print(f"Connected to MAIA Segmentation Portal: {maia_segmentation_portal_url}")
        model_metadata = response.json()["models"]["MONetBundle"]["metadata"]
        required_input_channels = model_metadata["inputs"]
        print(f"Required input channels: {len(required_input_channels)}")
        for idx, channel in enumerate(required_input_channels):
            print(f"Input Channel {idx}: {channel}")
        print(f"Expected Output Labels: {model_metadata['outputs']['pred']['channel_def']}")

    with open(input_path, "rb") as f:

        print("\n")
        print("DISCLAIMER: This tool is for research only. Authors disclaim responsibility for non-research use or outcomes.")
        print(
            "Files used for inference will be uploaded to an external server (MAIA Segmentation portal). Authors are not responsible"
        )
        print("for data handling, storage, or privacy on external platforms. Use at your own discretion.")
        print("\n")
        response = input("Are you sure you want to continue? (y/n): ")
        if response.lower() != "y":
            print("Operation cancelled.")
            return
        if input_path.endswith(".nii.gz"):
            print(f"Verifying input file: {input_path}")
            input_img = sitk.ReadImage(input_path)
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
                print(f"Input file {input_path} is valid for the model {model}.")
        files = {"params": (None, json.dumps(params), "application/json"), "file": (input_path, f, "application/gzip")}
        print(f"Sending request with input: {input_path}")
        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            with open(output_path, "wb") as out_file:
                out_file.write(response.content)
            print(f"Segmentation saved to: {output_path}")
        else:
            print(f"Request failed [{response.status_code}]: {response.text}")


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Run Remote segmentation inference on MAIA Segmentation Portal.")
    parser.add_argument("--input", "-i", required=True, help="Path to input .nii.gz file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the output segmentation")
    parser.add_argument("--username", required=True, help="Username for MAIA Segmentation Portal")
    # Parse username first to get available models
    if False:
        ...
        # temp_args, _ = parser.parse_known_args()
        # home = os.path.expanduser("~")
        # auth_path = os.path.join(home, ".monet", f"{temp_args.username}_auth.json")

        # with open(auth_path, "r") as token_file:
        #    token_data = json.load(token_file)
        #    token = token_data.get("access_token")
        #    models = get_available_models(token, temp_args.username)
    parser.add_argument("--model", required=True, help="Model to use for segmentation")

    return parser


def main():
    parser = get_arg_parser()

    args = parser.parse_args()
    run_inference(args.input, args.output, args.model, args.username)


if __name__ == "__main__":
    main()
