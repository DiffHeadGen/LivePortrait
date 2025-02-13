ml cuDNN/8.7.0.84-CUDA-11.8.0

# python inference.py

empty_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ', ' '{if ($2 < 512) print $1}' | head -n 1)
if [ -z "$empty_gpu" ]; then
    echo "No empty GPU available"
    exit 1
fi
CUDA_VISIBLE_DEVICES=$empty_gpu python infer.py

# CUDA_VISIBLE_DEVICES=$empty_gpu python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4
