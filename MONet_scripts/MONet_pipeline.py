#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow
import yaml
from monai.nvflare.utils import plan_and_preprocess_api, prepare_bundle_api, prepare_data_folder_api, train_api, validation_api


def pipeline(config):

    for step in config["steps"]:
        if step == "prepare":
            prepare_data_folder_api(
                data_dir=config["data_dir"],
                nnunet_root_dir=config["nnunet_root_dir"],
                dataset_name_or_id=config["dataset_name_or_id"],
                modality_dict=config["modality_dict"],
                output_data_dir=config["output_data_dir"],
                dataset_format=config["dataset_format"],
                experiment_name=config["experiment_name"],
                modality_list=(
                    config["modality_list"]
                    if "modality_list" in config
                    else [modality for modality in config["modality_dict"].keys() if modality != "label"]
                ),
                subfolder_suffix=None,
                patient_id_in_file_identifier=True,
                trainer_class_name="nnUNetTrainer",
                concatenate_modalities_flag=config["concat_modalities_flag"],
                regions_class_order=config["regions_class_order"] if "regions_class_order" in config else None,
                labels=config["labels"] if "labels" in config else None,
            )

        if step == "plan_and_preprocess":
            plan_and_preprocess_api(
                nnunet_root_dir=config["nnunet_root_dir"],
                dataset_name_or_id=config["dataset_name_or_id"],
                trainer_class_name="nnUNetTrainer",
                nnunet_plans_name="nnUNetPlans",
            )
        if step == "prepare_bundle":
            bundle_path = Path(config["bundle_config"]["bundle_root"]).parent
            subprocess.run(
                [
                    "wget",
                    "https://raw.githubusercontent.com/minnelab/MONet-Bundle/main/MONetBundle.zip",
                    "-O",
                    str(Path(bundle_path).joinpath("MONetBundle.zip")),
                ]
            )
            subprocess.run(["unzip", "-o", str(Path(bundle_path).joinpath("MONetBundle.zip")), "-d", str(bundle_path)])
            bundle_config = config["bundle_config"]
            bundle_config["dataset_name_or_id"] = config["dataset_name_or_id"]
            bundle_config["label_dict"] = config["label_dict"]
            if "trainer_class_name" in config:
                bundle_config["nnunet_trainer_class_name"] = config["trainer_class_name"]
            if "nnunet_plans_name" in config:
                bundle_config["nnunet_plans_identifier"] = config["nnunet_plans_name"]
            bundle_config["mlflow_experiment_name"] = config["experiment_name"]

            train_extra_configs = config.get("train_extra_configs", None)
            prepare_bundle_api(bundle_config=bundle_config, train_extra_configs=train_extra_configs)
        if step == "train":
            train_api(
                config["nnunet_root_dir"],
                dataset_name_or_id=config["dataset_name_or_id"],
                trainer_class_name="nnUNetTrainer",
                run_with_bundle=True,
                bundle_root=config["bundle_config"]["bundle_root"],
                continue_training=config["continue_training"] if "continue_training" in config else False,
                fold=0,
                experiment_name=config["experiment_name"],
                client_name=config["bundle_config"]["mlflow_run_name"][len("run_") :],
                tracking_uri=config["bundle_config"]["tracking_uri"],
                skip_training=config["run_validation_only"] if "run_validation_only" in config else False,
                resume_epoch="latest",
            )
        if step == "validate":
            validation_summary_dict, labels = validation_api(
                config["nnunet_root_dir"],
                config["dataset_name_or_id"],
                trainer_class_name="nnUNetTrainer",
                nnunet_plans_name="nnUNetPlans",
                fold=0,
                skip_prediction=True,
            )
            mlflow_token = None
            tracking_uri = config["bundle_config"]["tracking_uri"]
            experiment_name = config["experiment_name"]
            client_name = config["bundle_config"]["mlflow_run_name"][len("run_") :]
            if mlflow_token is not None:
                os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
            if tracking_uri is not None:
                mlflow.set_tracking_uri(tracking_uri)

            try:
                mlflow.create_experiment(experiment_name)
            except Exception as e:
                print(e)
                mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

            run_name = f"run_validation_{client_name}"

            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                order_by=["start_time DESC"],
            )

            if len(runs) == 0:
                with mlflow.start_run(run_name=f"run_validation_{client_name}", tags={"client": client_name}):
                    mlflow.log_dict(validation_summary_dict, "validation_summary.json")
                    for label in validation_summary_dict["mean"]:
                        for metric in validation_summary_dict["mean"][label]:
                            label_id = label
                            if "(" in label:
                                label_id = label.replace("(", "[").replace(")", "]")
                            if label_id not in labels:
                                print(f"Label {label_id} not found in labels dictionary. Skipping metric logging.")
                                continue
                            label_name = labels[label_id]
                            mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

            else:
                with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
                    mlflow.log_dict(validation_summary_dict, "validation_summary.json")
                    for label in validation_summary_dict["mean"]:
                        for metric in validation_summary_dict["mean"][label]:
                            label_id = label
                            if "(" in label:
                                label_id = label.replace("(", "[").replace(")", "]")
                            if label_id not in labels:
                                print(f"Label {label_id} not found in labels dictionary. Skipping metric logging.")
                                continue
                            label_name = labels[label_id]
                            mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline for nnUNet")
    parser.add_argument("--config", type=str, help="Path to the config file")
    
    return parser


def main():
    args = get_arg_parser().parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        pipeline(config)
    else:
        print("No config file provided")


if __name__ == "__main__":
    main()
