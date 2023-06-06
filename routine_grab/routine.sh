#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/c/Users/lizhe53/PycharmProjects/SmartTrade
source /mnt/c/lizhe53/AppData/Local/Continuum/anaconda3/Scripts/activate SmartTrade
routine_date=$(date +%Y-%m-%d)

nohup python routine_grab.py $routine_date > log/log_${routine_date}.log 2>&1 &
