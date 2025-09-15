#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pydicom
import requests
import SimpleITK as sitk
from pynetdicom.apps.storescu.storescu import main as storescu_main

from MONet.utils import get_available_models


def run_dicom_inference(input_path, output_path, model, username):
    # Find the first .dcm file in input_path or its subdirectories using pydicom
    dcm_file = None
    studyInstanceUID = None
    if "DICOM_URL" in os.environ:
        dicom_url = os.environ["DICOM_URL"]
    else:
        dicom_url = "localhost"
    if not Path(input_path).is_dir():
        studyInstanceUID = input_path

    if studyInstanceUID is None:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(".dcm"):
                    dcm_file = os.path.join(root, file)
                    break
            if dcm_file:
                break
        if not dcm_file:
            raise FileNotFoundError(f"No .dcm file found in {input_path} or its subdirectories.")
            return
        ds = pydicom.dcmread(dcm_file)
        studyInstanceUID = ds.get("StudyInstanceUID", "Unknown")
        print(f"Study Instance UID: {studyInstanceUID}")
        print(f"Patient ID: {ds.get('PatientID', 'Unknown')}")
        storescu_main(["storescu", "-r", dicom_url, "4242", input_path])
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

    allowed_label_names = ["spleen"]
    # Retrieve labels using requests
    info_response = requests.get(f"{maia_segmentation_portal_url}info/", headers=headers)
    if info_response.status_code == 200:
        labels = info_response.json()["models"]["MONetBundle"].get("labels", {})
        labels_info = []
        for i, label in enumerate(labels):
            label_name = label
            if label_name not in allowed_label_names:
                label_name = f"region {i}"
            model_name = "MAIA-Segmentation-Portal"
            labels_info.append({"name": label_name, "model_name": model_name})
    else:
        print(f"Failed to retrieve labels [{info_response.status_code}]: {info_response.text}")
        return
    params = {
        # 'device': 'NVIDIA GeForce RTX 2070 SUPER:0',
        # 'model_filename': 'model.ts',
        "image": studyInstanceUID,  # Use StudyInstanceUID as the image identifier
        "output": "dicom_seg",
    }
    params_str = '{"label_info":['
    for idx, label in enumerate(labels_info):
        params_str += f'{{"name":"{label["name"]}","model_name":"{label["model_name"]}"}}'
        if idx < len(labels_info) - 1:
            params_str += ","
    params_str += "]}"  # must be a JSON string
    form_data = {"params": params_str}
    print(f"Sending request with input: {input_path}")
    response = requests.post(url, headers=headers, params=params, data=form_data)

    if response.status_code == 200:
        with open(output_path, "wb") as out_file:
            out_file.write(response.content)
        print(f"Segmentation saved to: {output_path}")
    else:
        print(f"Request failed [{response.status_code}]: {response.text}")
    # You can now process dcm_files as needed
    storescu_main(["storescu", dicom_url, "4242", output_path])


def run_inference(input_path, output_path, model, username):
    if Path(input_path).is_dir():
        run_dicom_inference(input_path, output_path, model, username)
        return
    elif not Path(input_path).is_file():
        run_dicom_inference(input_path, output_path, model, username)
        return
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
