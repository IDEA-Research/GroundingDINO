#!/bin/bash
export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib/python3.10/site-packages/torch/lib:/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
source /home/cluster/miniforge3/etc/profile.d/conda.sh && conda activate webapp_mongo_gpu && python web_app.py