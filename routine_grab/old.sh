#!/bin/bash
# coding: utf-8

routine_date=$(date +%Y-%m-%d)
rd='seas'

nohup python routine_grab.py routine_date > log/log_$rd 2>&1 &
