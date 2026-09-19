#!/bin/bash
python3 main.py configs/sample_nms.json
python3 main.py configs/sample_nms_joint.json
python3 main.py configs/sample_spa.json

# Generate plots, statistics, and animations for each training history.
python3 output/postproc.py output/history_nms
python3 output/postproc.py output/history_nms_joint
python3 output/postproc.py output/history_spa
