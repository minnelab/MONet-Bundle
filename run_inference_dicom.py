from python_on_whales import docker

docker.run(
        "lymphoma-x64-workstation-dgpu-linux-amd64:1.0",
        gpus="device=0",
        volumes=[
            ("<DICOM_STUDY_FOLDER>", "/var/holoscan/input"),
            ("<FEDERATED_LYMPHOMA_TORCHSCRIPT_MODEL>", "/opt/holoscan/models"),
            ("<PREDICTION_OUTPUT_FOLDER>", "/var/holoscan/output")],
        shm_size="2g",
    )