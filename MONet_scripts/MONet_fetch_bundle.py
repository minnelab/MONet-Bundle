#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


def get_arg_parser():

    parser = argparse.ArgumentParser(description="Fetch MONet Bundle and extract it to the specified path.")
    parser.add_argument("--bundle_path", type=str, required=True, help="Path to save the MONet Bundle.")
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    url = "https://raw.githubusercontent.com/minnelab/MONet-Bundle/main/MONetBundle.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(Path(args.bundle_path).joinpath("MONetBundle.zip"), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded MONet Bundle to {args.bundle_path}/MONetBundle.zip")
    print("Extracting MONet Bundle...")
    with zipfile.ZipFile(Path(args.bundle_path).joinpath("MONetBundle.zip"), "r") as zip_ref:
        zip_ref.extractall(args.bundle_path)
    print(f"MONet Bundle extracted to {args.bundle_path}")


if __name__ == "__main__":
    main()
