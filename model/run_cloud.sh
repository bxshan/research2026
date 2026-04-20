#!/bin/bash
# Run GT then PS training sequentially, logging output to both terminal and file.

LOG=/workspace/model/logs/run_cloud_$(date +%Y%m%d_%H%M%S).txt
CFG=/workspace/model/cfgs/train_config_cloud.yaml
SCRIPT=/workspace/model/sft_bias.py

echo "Logging to $LOG"

python3 $SCRIPT --dataset gt --config $CFG --steps 5000 2>&1 | tee -a $LOG && \
python3 $SCRIPT --dataset ps --config $CFG --steps 5000 2>&1 | tee -a $LOG
