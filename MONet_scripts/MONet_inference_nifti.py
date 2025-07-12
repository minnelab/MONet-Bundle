#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

try:
    from python_on_whales import docker
except ImportError:
    docker = None


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run MONet inference on NIFTI data using a TorchScript model.")
    parser.add_argument("--study_folder", type=str, required=True, help="Path to the NIFTI study folder.")
    parser.add_argument("--docker-image", type=str, required=True, help="Docker image to use for inference.")
    parser.add_argument("--prediction_output_folder", type=str, required=True, help="Path to save the prediction outputs.")
    parser.add_argument(
        "--multi-folder", action="store_true", help="If set, the script will process multiple folders within the study folder."
    )
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    Path(args.prediction_output_folder).mkdir(parents=True, exist_ok=True)
    if args.multi_folder:
        # If multi-folder is set, process each subfolder in the study folder
        study_folders = [f for f in Path(args.study_folder).iterdir() if f.is_dir()]
        for study_folder in study_folders:
            Path(args.prediction_output_folder).mkdir(parents=True, exist_ok=True)
            docker.run(
                image=args.docker_image,
                gpus="device=0",
                volumes=[(study_folder, "/var/holoscan/input"), (args.prediction_output_folder, "/var/holoscan/output")],
                shm_size="2g",
            )
    else:

        docker.run(
            image=args.docker_image,
            gpus="device=0",
            volumes=[
                (args.study_folder, "/var/holoscan/input"),
                # (torchscript_model, "/opt/holoscan/models"),
                (args.prediction_output_folder, "/var/holoscan/output"),
            ],
            shm_size="2g",
        )


if __name__ == "__main__":
    main()
