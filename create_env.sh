ml cuDNN/8.7.0.84-CUDA-11.8.0

conda deactivate
# conda env remove -n liveportrait -y
conda create -n liveportrait python=3.10 -y
conda activate liveportrait
conda install -c conda-forge ffmpeg -y
# pip3 install torch torchvision torchaudio
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install onnxruntime-gpu==1.18.0

pip install -e ../expdata

huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
