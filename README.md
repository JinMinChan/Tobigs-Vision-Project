# Tobigs-Vision-Project
[GroundedSam2](GroundedSam2.md) | [Size_Estimation](Size_Estimation.md)
<p align="center">
  <img src="Conference_Poster.jpg" alt="poster" width="45%">
</p>
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
Demo Script
```python
#지정된 이미지 및 참조 물체로 결과 추출
python tobigs.py
#Gradio를 통해 이미지, 원하는 가구, 참조물체, 참조물체 크기를 Input으로 줄 수 있음
python tobigs_gradio.py
```
