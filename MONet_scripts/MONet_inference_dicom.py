#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

try:
    from python_on_whales import docker
except ImportError:
    docker = None


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run MONet inference on DICOM data using a TorchScript model.")
    parser.add_argument("--dicom_study_folder", type=str, required=True, help="Path to the DICOM study folder.")
    parser.add_argument("--docker-image", type=str, required=True, help="Docker image to use for inference.")
    parser.add_argument("--prediction_output_folder", type=str, required=True, help="Path to save the prediction outputs.")
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    Path(args.prediction_output_folder).mkdir(parents=True, exist_ok=True)
    docker.run(
        image=args.docker_image,
        gpus="device=0",
        volumes=[
            (args.dicom_study_folder, "/var/holoscan/input"),
            # (torchscript_model, "/opt/holoscan/models"),
            (args.prediction_output_folder, "/var/holoscan/output"),
        ],
        shm_size="2g",
    )


if __name__ == "__main__":
    main()
