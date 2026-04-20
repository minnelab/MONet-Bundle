#!/bin/bash -e

if [ "$1" == "convert_nnunet_to_monet" ]; then
    MONet_convert_nnunet_checkpoint_to_bundle_checkpoint "${@:2}"
    MONet_convert_ckpt_to_ts "${@:2}"
else
    "$@"
fi