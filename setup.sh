#!/bin/bash
nohup python3 -u run.py >./log/train5.log 2>&1 &
tail -f ./log/train5.log