from python_on_whales import docker
from pathlib import Path
dicom_study_folder = "<DICOM_STUDY_FOLDER>"  # Replace with the actual DICOM study folder path
torchscript_model = "<SPLEEN_TORCHSCRIPT_MODEL>"  # Replace with the actual TorchScript model path
prediction_output_folder = "<PREDICTION_OUTPUT_FOLDER>"  # Replace with the actual output folder path

Path(prediction_output_folder).mkdir(parents=True, exist_ok=True)
docker.run(
        "spleen-x64-workstation-dgpu-linux-amd64:1.0",
        gpus="device=0",
        volumes=[
            (dicom_study_folder, "/var/holoscan/input"),
            (torchscript_model, "/opt/holoscan/models"),
            (prediction_output_folder, "/var/holoscan/output")],
        shm_size="2g",
    )