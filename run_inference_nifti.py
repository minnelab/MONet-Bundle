from python_on_whales import docker

docker.run(
        "spleen-x64-workstation-dgpu-linux-amd64:1.0",
        gpus="device=0",
        volumes=[
            ("./imagesTs", "/var/holoscan/input"),
            ("./predsTs", "/var/holoscan/output")],
        shm_size="2g",
    )