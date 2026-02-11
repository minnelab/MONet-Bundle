import os
import yaml
import re
import torch
import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Convert Lightning checkpoint to PyTorch checkpoint")
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the config file")
    return parser


def main():
    args = get_arg_parser().parse_args()
    config_file_path = args.config_file_path
    with open(config_file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    ckpt_dir = os.path.join(config_dict["bundle_config"]["bundle_root"], "models", "fold_{}".format(config_dict.get("fold", 0)))
    files = os.listdir(ckpt_dir)

    ckpt_files = [f for f in files if f.endswith('.ckpt')]

    for ckpt_file in ckpt_files:
        match = re.match(r'epoch=(\d+)-([a-zA-Z0-9_]+)=([0-9\.]+)\.ckpt', ckpt_file)
        if match:
            print(f"{ckpt_file} matches: epoch={match.group(1)}, metric={match.group(2)}, value={match.group(3)}")
            state_dict = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=torch.device('cpu'))
            new_state_dict = {}
            new_state_dict["network_weights"] = {}
            for k, v in state_dict['state_dict'].items():
                new_key = k.replace('network._orig_mod.', '')
                new_state_dict["network_weights"][new_key] = v
            new_state_dict["optimizer_state"] = state_dict['optimizer_states'][0]
            new_state_dict["scheduler"] = state_dict['lr_schedulers'][0]
            val_metric = match.group(3)
            torch.save(new_state_dict, os.path.join(ckpt_dir, f"checkpoint_key_metric={val_metric}.pt"))
        else:
            print(f"{ckpt_file} does not match the pattern.")
        match = re.match(r'epoch=(\d+)\.ckpt', ckpt_file)
        if match:
            print(f"{ckpt_file} matches: epoch={match.group(1)}")
            state_dict = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=torch.device('cpu'))
            new_state_dict = {}
            new_state_dict["network_weights"] = {}
            for k, v in state_dict['state_dict'].items():
                new_key = k.replace('network._orig_mod.', '')
                new_state_dict["network_weights"][new_key] = v
            new_state_dict["optimizer_state"] = state_dict['optimizer_states'][0]
            new_state_dict["scheduler"] = state_dict['lr_schedulers'][0]
            epoch_num = int(match.group(1)) + 1
            torch.save(new_state_dict, os.path.join(ckpt_dir, f"checkpoint_epoch={epoch_num}.pt"))
        else:
            print(f"{ckpt_file} does not match the pattern.")


if __name__ == "__main__":
    main()
