#!/bin/bash -e


MONet_convert_nnunet_checkpoint_to_bundle_checkpoint "$@"
MONet_convert_ckpt_to_ts "$@"