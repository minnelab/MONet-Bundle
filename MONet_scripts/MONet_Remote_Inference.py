#!/usr/bin/env python3

import argparse
import requests
import json
from MONet.utils import get_available_models
import os
import sys

def run_inference(input_path, output_path, model):

    home = os.path.expanduser("~")
    auth_path = os.path.join(home, ".monet", f"{args.username}_auth.json")
    with open(auth_path, "r") as token_file:
        token_data = json.load(token_file)
        token = token_data.get('access_token')
        models = get_available_models(token, args.username)
        maia_segmentation_portal_url = models.get(model)
        if not maia_segmentation_portal_url:
            raise ValueError(f"Model '{model}' is not supported. Available models: {list(models.keys())}")
        url = f'{maia_segmentation_portal_url}infer/MONetBundle?output=image'
        if not token:
            raise ValueError("Access token not found in the token file.")
    
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {token}',  # Replace with your actual token
    }

    params = {
        # 'device': 'NVIDIA GeForce RTX 2070 SUPER:0',
        # 'model_filename': 'model.ts',
    }

    with open(input_path, 'rb') as f:
        files = {
            'params': (None, json.dumps(params), 'application/json'),
            'file': (input_path, f, 'application/gzip'),
        }
        print("\n")
        print("DISCLAIMER: This tool is provided solely for research purposes. The authors and contributors disclaim any responsibility for its use outside of research contexts or for any outcomes resulting from its use. Please be aware that any files used for inference with this script will be uploaded to an external server—the MAIA segmentation portal. The authors do not assume any responsibility for data handling, storage, or privacy on external platforms. Use at your own discretion.")
        print("\n")
        response = input("Are you sure you want to continue? (y/n): ")
        if response.lower() != "y":
            print("Operation cancelled.")
            return
        print(f"Sending request with input: {input_path}")
        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.content)
            print(f"Segmentation saved to: {output_path}")
        else:
            print(f"Request failed [{response.status_code}]: {response.text}")

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Run Remote segmentation inference on MAIA Segmentation Portal.')
    parser.add_argument('--input', '-i', required=True, help='Path to input .nii.gz file')
    parser.add_argument('--output', '-o', required=True, help='Path to save the output segmentation')
    parser.add_argument('--username', required=True, help='Username for MAIA Segmentation Portal')
    # Parse username first to get available models
    if False:
        temp_args, _ = parser.parse_known_args()
        home = os.path.expanduser("~")
        auth_path = os.path.join(home, ".monet", f"{temp_args.username}_auth.json")

        with open(auth_path, "r") as token_file:
            token_data = json.load(token_file)
            token = token_data.get('access_token')
            models = get_available_models(token, temp_args.username)
        parser.add_argument('--model', required=True, choices=models.keys(), help='Model to use for segmentation')

    return parser

def main():
    parser = get_arg_parser()

    args = parser.parse_args()
    run_inference(args.input, args.output, args.model)

if __name__ == '__main__':
    main()    

