PKG_DIR=~/workspace/libs
#export CUDA_HOME=/usr/local/cuda #11.4
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PKG_DIR}/TensorRT-8.5.3.1_cuda11/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PKG_DIR}/cudnn-linux-8.9.6.50_cuda11/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/targets/aarch64-linux/lib
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PKG_DIR}/TensorRT-8.6.1.6_cuda12/lib:${PKG_DIR}/cudnn-linux-x86_64-8.9.6.50_cuda12/lib:${CUDA_HOME}/targets/aarch64-linux/lib
PYTHONPATH=$PYTHONPATH:../ python3 convert_tensorrt.py -m ./grounding_dino_sim_op11.onnx -o grounding_dino.trtengine --use_fp16
