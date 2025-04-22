#!/bin/bash -e


/workspace/convert_nnunet_checkpoint_to_bundle_checkpoint.py "$@"
/workspace/convert_ckpt_to_ts.py "$@"