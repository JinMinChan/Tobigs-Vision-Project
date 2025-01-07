# Tobigs-Vision-Project

## Environment
- Ubuntu 20.04
- Python 3.10.6
- Cuda 12.1
  
가상환경 생성:
```python
conda create --name tobigs python=3.10 -y
```
PyTorch 및 TorchVision 종속성 설치
```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Segment Anything 2 설치
```python
pip install -e .
```
Grounding DINO 설치
```python
pip install --no-build-isolation -e grounding_dino
```
추가로 설치해야 할 라이브러리
```python
pip install opencv-python supervision pycocotools transformers addict yapf timm gradio
```
SAM 2 사전 학습 체크포인트
```python
cd checkpoints
bash download_ckpts.sh
```
Grounding Dino 사전 학습 체크포인트
```python
cd gdino_checkpoints
bash download_ckpts.sh
```
