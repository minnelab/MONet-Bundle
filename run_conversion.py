from python_on_whales import docker
import zipfile
import os
# Ensure the nnUNet_Models folder exists

if __name__ == "__main__":
    
    bundle_path = "./nnUNetBundle"
    
    os.makedirs("nnUNet_Models", exist_ok=True)

    # Unzip the nnUNet bundle into the nnUNet_Models folder
    with zipfile.ZipFile("Task09_Spleen.zip", 'r') as zip_ref:
        zip_ref.extractall("nnUNet_Models")

    docker.run(
        "nnunet_monai_converter",
        gpus="device=0",
        envs={
            "nnUNet_results": "/input",

        },
        command=[
            "--fold", "0", "--bundle_root", "/model/bundle","--dataset_name_or_id","09"
            ],
        volumes=[
            (bundle_path, "/model/bundle"),
            ("./nnUNet_Models", "/input")],
        shm_size="2g",
    )
    
    model_folder = os.path.join(bundle_path, "models", "fold_0")
    
    print("Conversion completed successfully.")
    print(f"Please check the folder {bundle_path} for the converted nnUNet model in the MONAI Bundle format.")
    print(f"You can find the TorchScript model in the {model_folder} folder.")
