from python_on_whales import docker
from pathlib import Path

study_folder = "<STUDY_FOLDER>"  # Replace with the actual study folder path
checkpoints_folder = "<CHECKPOINTS_FOLDER>"  # Replace with the actual checkpoints folder path
predictions_folder = "<PREDICTIONS_FOLDER>"  # Replace with the actual predictions folder

Path(predictions_folder).mkdir(parents=True, exist_ok=True)

docker.run(
        "spleen-x64-workstation-dgpu-linux-amd64:1.0",
        gpus="device=0",
        volumes=[
            (study_folder, "/var/holoscan/input"),
            (checkpoints_folder, "/opt/holoscan/models"),
            (predictions_folder, "/var/holoscan/output")],
        shm_size="2g",
    )