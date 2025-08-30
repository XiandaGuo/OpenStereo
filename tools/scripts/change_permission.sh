#!/bin/bash

source tools/scripts/constant.sh

save_root_dir="${CKPT_ROOT_DIR%/}/$1"

useradd -u 1002 xianda.guo
chown -R xianda.guo:xianda.guo $save_root_dir
