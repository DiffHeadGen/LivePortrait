module load cuDNN/8.7.0.84-CUDA-11.8.0

gpu_indices=(0 1 2 3 4 5 6 7)
mkdir -p logs
for gpu in "${gpu_indices[@]}"
do
    echo "Starting instance on GPU $gpu"
    sleep 2
    CUDA_VISIBLE_DEVICES=$gpu python infer.py > logs/log_$gpu.txt 2>&1 &
done

# nvidia-smi | grep 'python' | awk '{print $5}' | xargs -I{} kill -9 {}
