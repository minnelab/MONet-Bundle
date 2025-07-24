from __future__ import annotations

import argparse
import os
import shutil
import socket
from pathlib import Path

import monai
import pytorch_lightning as L
import src
import torch
import yaml
from monai.bundle import ConfigParser
from monai.transforms import Decollated
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from src.trainer import prepare_nnunet_batch, subfiles


def get_lightning_checkpoint(epoch, ckpt_dir):
    if epoch == "latest":

        latest_checkpoints = subfiles(ckpt_dir, prefix="epoch=", sort=True, join=False)
        epochs = []
        for latest_checkpoint in latest_checkpoints:
            try:
                epochs.append(int(latest_checkpoint[len("epoch=") : -len(".ckpt")]))
            except Exception as e:
                print(f"Error parsing checkpoint '{latest_checkpoint}': {e}")

        epochs.sort()
        latest_epoch = epochs[-1]
        return latest_epoch
    else:
        return epoch


class MONetBundleDataModule(L.LightningDataModule):
    def __init__(self, yaml_config_files, override=None):
        super().__init__()
        self.yaml_config_files = yaml_config_files
        self.parser = None
        self.override = override

    def prepare_data(self): ...
    def setup(self, stage: str):
        monai_config = {}
        for config_file in self.yaml_config_files:
            with open(config_file, "r") as file:
                monai_config.update(yaml.safe_load(file))

        if self.override:
            for override_key in self.override:
                monai_config[override_key] = self.override[override_key]

        self.parser = ConfigParser(monai_config, globals={"os": "os", "pathlib": "pathlib", "json": "json", "ignite": "ignite"})

        self.parser.parse(True)
        self.train_loader = self.parser.get_parsed_content("train#dataloader", instantiate=True)
        self.val_loader = self.parser.get_parsed_content("validate#dataloader", instantiate=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self): ...

    def predict_dataloader(self): ...

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


class MONetBundleModule(L.LightningModule):
    def __init__(self, yaml_config_files, override=None):
        super().__init__()
        self.automatic_optimization = False
        monai_config = {}
        for config_file in yaml_config_files:
            with open(config_file, "r") as file:
                monai_config.update(yaml.safe_load(file))
        self.override = override
        if self.override:
            for override_key in self.override:
                monai_config[override_key] = self.override[override_key]

        self.parser = ConfigParser(monai_config, globals={"os": "os", "pathlib": "pathlib", "json": "json", "ignite": "ignite"})

        self.parser.parse(True)
        self.network = self.parser.get_parsed_content("network", instantiate=True)
        self.loss = self.parser.get_parsed_content("loss", instantiate=True)
        self.prepare_batch = prepare_nnunet_batch

        self.optimizer = self.parser.get_parsed_content("optimizer", instantiate=True)
        try:
            self.scheduler = self.parser.get_parsed_content("lr_scheduler", instantiate=True)
        except KeyError as e:
            self.scheduler = None
            print(f"Error parsing lr_scheduler: {e}")

        self.train_key_metric = self.parser.get_parsed_content("train_key_metric", instantiate=True)["Train_Dice"]
        self.val_key_metric = self.parser.get_parsed_content("val_key_metric", instantiate=True)["Val_Dice"]

        self.train_additional_metrics = self.parser.get_parsed_content("train_additional_metrics", instantiate=True)
        self.val_additional_metrics = self.parser.get_parsed_content("val_additional_metrics", instantiate=True)

        self.output_transform = monai.handlers.from_engine(["pred", "label"])

        self.decollate_transform = Decollated(keys=None, detach=True)
        self.postprocessing = monai.transforms.Compose(
            [self.decollate_transform] + list(self.parser.get_parsed_content("train_postprocessing", instantiate=True).transforms)
        )

        try:
            self.region_based = self.parser.get_parsed_content("region_based", instantiate=True)
        except KeyError:
            self.region_based = False

        if self.region_based:
            self.postprocessing = monai.transforms.Compose(
                [self.decollate_transform]
                + list(self.parser.get_parsed_content("train_postprocessing_region_based", instantiate=True).transforms)
            )

        try:
            self.num_train_batches_per_epoch = self.parser.get_parsed_content("iterations", instantiate=True)
        except KeyError as e:
            self.num_train_batches_per_epoch = None
            print(f"Error parsing iterations: {e}")

        try:
            self.num_val_batches_per_epoch = self.parser.get_parsed_content(
                "nnunet_trainer.num_val_iterations_per_epoch", instantiate=True
            )
        except KeyError:
            self.num_val_batches_per_epoch = None
        self.max_num_epochs = self.parser.get_parsed_content("epochs", instantiate=True)

        self.mlflow_tracking_uri = self.parser.get_parsed_content("tracking_uri", instantiate=True)
        self.experiment_name = self.parser.get_parsed_content("mlflow_experiment_name", instantiate=True)
        self.run_name = self.parser.get_parsed_content("mlflow_run_name", instantiate=True)
        self.ckpt_dir = self.parser.get_parsed_content("ckpt_dir", instantiate=True)

        self.trainer = self.parser.get_parsed_content("train#trainer", instantiate=True)
        self.evaluator = self.parser.get_parsed_content("validate#evaluator", instantiate=True)
        self.val_key_metric.attach(self.evaluator, "val_key_metric")
        self.train_key_metric.attach(self.trainer, "train_key_metric")

        self.experiment_hyperparams = src.mlflow.create_mlflow_experiment_params(
            monai_config["bundle_root"] + "/nnUNet/params.yaml"
        )

        self.experiment_hyperparams.update(
            {
                "dataset_name_or_id": self.parser.get_parsed_content("dataset_name_or_id", instantiate=True),
                "fold_id": self.parser.get_parsed_content("fold_id", instantiate=True),
                "iterations": self.num_train_batches_per_epoch,
                "epochs": self.max_num_epochs,
                "label_dict": self.parser.get_parsed_content("label_dict", instantiate=True),
                "mlflow_experiment_name": self.experiment_name,
                "mlflow_run_name": self.run_name,
                "tracking_uri": self.mlflow_tracking_uri,
                "nnunet_plans_identifier": self.parser.get_parsed_content("nnunet_plans_identifier", instantiate=True),
                "nnunet_trainer_class_name": self.parser.get_parsed_content("nnunet_trainer_class_name", instantiate=True),
                "num_classes": self.parser.get_parsed_content("num_classes", instantiate=True),
                "nnunet_configuration": self.parser.get_parsed_content("nnunet_configuration", instantiate=True),
            }
        )

        self.checkpoint = self.parser.get_parsed_content("checkpoint", instantiate=True)
        self.checkpoint_filename = self.parser.get_parsed_content("checkpoint_filename", instantiate=True)
        self.nnunet_model_folder = self.parser.get_parsed_content("nnunet_model_folder", instantiate=True)
        self.bundle_root = self.parser.get_parsed_content("bundle_root", instantiate=True)

        torch.save(self.checkpoint, self.checkpoint_filename)
        shutil.copy(Path(self.nnunet_model_folder).joinpath("dataset.json"), self.bundle_root + "/models/dataset.json")
        shutil.copy(Path(self.nnunet_model_folder).joinpath("plans.json"), self.bundle_root + "/models/plans.json")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        inputs, targets = self.prepare_batch(batch, self.device, True)

        outputs = self.network(inputs)

        loss = self.loss(outputs, targets)

        self.train_key_metric.update(self.output_transform(self.postprocessing({"pred": outputs, "label": targets})))
        for metric in self.train_additional_metrics:
            self.train_additional_metrics[metric].update(
                self.output_transform(self.postprocessing({"pred": outputs, "label": targets}))
            )

        if loss is not None:
            self.manual_backward(loss)
            opt.step()
            self.log("Train_Loss", loss, sync_dist=True)
            return loss
        else:
            return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch, self.device, True)
        outputs = self.network(inputs)
        self.val_key_metric.update(self.output_transform(self.postprocessing({"pred": outputs, "label": targets})))
        for metric in self.val_additional_metrics:
            self.val_additional_metrics[metric].update(
                self.output_transform(self.postprocessing({"pred": outputs, "label": targets}))
            )

    def on_train_epoch_end(self):
        key_train_metric = self.train_key_metric.compute()
        self.train_key_metric.reset()

        additional_metrics = {}
        for metric in self.train_additional_metrics:
            additional_metrics[metric] = self.train_additional_metrics[metric].compute()
            self.train_additional_metrics[metric].reset()

        # self.logger.log_metrics(key_val_metric, self.current_epoch)

        self.log("Train_Dice", key_train_metric, sync_dist=True)
        for metric in self.train_additional_metrics:
            self.log(metric, additional_metrics[metric], sync_dist=True)

        return key_train_metric

    def on_validation_epoch_end(self):
        key_val_metric = self.val_key_metric.compute()
        self.val_key_metric.reset()

        additional_metrics = {}
        for metric in self.val_additional_metrics:
            additional_metrics[metric] = self.val_additional_metrics[metric].compute()
            self.val_additional_metrics[metric].reset()
        # self.logger.log_metrics(key_val_metric, self.current_epoch)

        self.log("Val_Dice", key_val_metric, sync_dist=True)
        for metric in self.val_additional_metrics:
            self.log(metric, additional_metrics[metric], sync_dist=True)
        return key_val_metric

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]


def loggers_and_callbacks(model):
    pl_logger = MLFlowLogger(
        experiment_name=model.experiment_name,
        tracking_uri=model.mlflow_tracking_uri,
        run_name="MONetBundle",
        tags={
            "host": socket.gethostname(),
            "fold": "0",
            "task": model.run_name,
            "job_id": os.environ["SLURM_JOB_ID"],
            "mlflow.runName": model.run_name,
        },
    )

    pl_logger.log_hyperparams(model.experiment_hyperparams)

    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=model.ckpt_dir, filename="{epoch}-{Val_Dice:.2f}", save_last=True, save_top_k=1, monitor="Val_Dice", mode="max"
    )
    checkpoint_cb.CHECKPOINT_NAME_LAST = "{epoch}"
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    plugins = None

    return pl_logger, callbacks, plugins


def get_arg_parser():
    parser = argparse.ArgumentParser(description="A Script to run Object Detection on PyTorch Lightning using a MONAI Bundle.")

    # Argument for a single file path
    parser.add_argument("--custom-params", type=str, required=False, help="Path to the custom parameters YAML file.")

    # Argument for a list of configuration files
    parser.add_argument("--config-files", type=str, nargs="+", required=True, help="List of Bundle configuration files.")

    # Optional boolean flag
    parser.add_argument(
        "--resume-from-latest-checkpoint",
        action="store_true",
        help="Flag to resume from the latest checkpoint. Default is False.",
    )

    return parser


def get_max_gpus():
    """Returns the number of available GPUs."""
    return torch.cuda.device_count()


def main():
    args = get_arg_parser().parse_args()
    if args.custom_params:
        with open(args.custom_params) as f:
            ov = yaml.safe_load(f)
    else:
        ov = None
    dm = MONetBundleDataModule(args.config_files, ov)
    model = MONetBundleModule(args.config_files, ov)

    logger, callbacks, plugins = loggers_and_callbacks(model)

    max_gpus = get_max_gpus()

    print(f"\nDetected {max_gpus} GPU(s).")

    # strategy = "ddp" if max_gpus > 1 else None

    num_train_batches_per_epoch = model.num_train_batches_per_epoch
    if model.num_train_batches_per_epoch is None:
        dm.prepare_data()
        dm.setup("train")
        num_train_batches_per_epoch = 250

    num_val_batches_per_epoch = model.num_val_batches_per_epoch
    if model.num_val_batches_per_epoch is None:
        dm.prepare_data()
        dm.setup("val")
        num_val_batches_per_epoch = 50

    trainer = L.Trainer(
        devices=max_gpus,  # Automatically detect GPUs or CPUs
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true",
        precision="16" if torch.cuda.is_available() else "32",  # Mixed precision for GPUs, FP32 for CPU
        # amp_backend=cfg["trainer_cfg"]["amp_backend"],
        # amp_level=cfg["trainer_cfg"]["amp_level"],
        # benchmark=cfg["trainer_cfg"]["benchmark"],
        # deterministic=cfg["trainer_cfg"]["deterministic"],
        callbacks=callbacks,
        logger=logger,
        max_epochs=model.max_num_epochs,
        num_sanity_val_steps=10,
        plugins=plugins,
        log_every_n_steps=num_train_batches_per_epoch,
        limit_train_batches=num_train_batches_per_epoch,
        limit_val_batches=num_val_batches_per_epoch,
    )

    print(
        f"Trainer initialized with devices={trainer.device_ids}, accelerator={trainer.accelerator}, strategy={trainer.strategy}, precision={trainer.precision}."
    )

    if args.resume_from_latest_checkpoint:
        latest_ckpt = get_lightning_checkpoint("latest", model.ckpt_dir)
        if latest_ckpt:
            trainer.fit(model, datamodule=dm, ckpt_path=model.ckpt_dir + f"/epoch={latest_ckpt}.ckpt")
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
