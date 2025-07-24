#!/usr/bin/env python3

from __future__ import annotations

import os

from python_on_whales import docker


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run a MONAI Label server with a TorchScript model.")
    parser.add_argument("--image_folder", type=str, required=False, help="Path to the image folder to be processed.")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the model folder.")
    parser.add_argument("--docker-image", type=str, required=True, help="Docker image to use for inference.")
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.isabs(args.model_folder):
        args.model_folder = os.path.abspath(args.model_folder)
    mount_volumes = [(args.model_folder, "/opt/holoscan/monailabel/monaibundle/model/MONetBundle/models/fold_0")]
    if args.image_folder:
        mount_volumes.append((args.image_folder, "/var/holoscan/input"))
    docker.run(
        image=args.docker_image,
        entrypoint="bash",
        command=[
            "monailabel",
            "start_server",
            "--app",
            "/opt/holoscan/monailabel/monaibundle",
            "--studies",
            "/var/holoscan/input",
            "--conf",
            "models",
            "MONetBundle",
        ],
        publish=[("8000", "8000")],
        gpus="device=0",
        volumes=mount_volumes,
        shm_size="2g",
    )


if __name__ == "__main__":
    main()
