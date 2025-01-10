# Segmentation을 활용한 물체 크기 및 거리 추정

이 프로젝트는 **Segmentation 마스크와 MiDaS 상대 깊이 맵**을 활용하여, 물체의 2D 크기(픽셀 단위)와 이를 바탕으로 실제 거리/크기를 추정하는 방법을 적용합니다.  
특히, Contour(윤곽선)와 RotatedRect(기울어진 사각형)을 사용하여 물체의 폭·높이를 정밀히 측정하는 방식을 활용합니다.

---

## 개요

### 기존 방식
- 단순히 축에 평행한 Bounding Box(BBox)를 구해 폭·높이를 측정.  
- 마스크가 기울어져 있으면 **정확도가 떨어짐**.

### 개선 방식
- Contour(윤곽선)를 찾아 OpenCV `cv2.minAreaRect()`로 기울어진 사각형(RotatedRect)을 생성.  
  - 마스크가 기울어져 있어도, 2D 상에서 가장 긴 축(width)과 그에 수직인 축(height)을 정확히 계산 가능.  

---

## 주요 과정

1. **Segmentation 마스크 처리**
   - `measure_2d_size_from_mask()`를 통해 마스크로부터 Contour를 찾고, RotatedRect로 2D 크기(픽셀 단위 폭·높이)를 측정.

2. **참조 물체 세그멘테이션**
   - 마스크로부터 참조물체의 2D 크기와 MiDaS 상대 깊이 맵의 평균값 계산.  
   - **핀홀 카메라 공식**을 사용해 참조 물체까지 거리와 전역 스케일 추정:
     \[
     s = \frac{Z_\mathrm{object}}{\bar{d}_\mathrm{ref}}
     \]

3. **장면 전체 깊이 계산**
   - 전역 스케일을 적용해 **절대 깊이 맵** 생성:
     \[
     \text{real\_depth\_map} = \text{relative\_depth\_map} \times s
     \]

4. **측정 대상 세그멘테이션**
   - 텀블러의 2D 크기(픽셀 단위 폭·높이)와 절대 깊이 맵을 결합하여 실제 크기 추정:
     \[
     \text{Width} = \frac{\text{픽셀 폭} \times \bar{Z}_\mathrm{tumbler} \times \text{센서 폭}}
                        {f \times \text{이미지 폭}}
     \]

---

## 주요 함수 설명

### 1. `measure_2d_size_from_mask(mask)`
- 입력: 0/1 이진화 마스크  
- 처리:
  - OpenCV `cv2.findContours()`로 마스크에서 윤곽선을 추출.  
  - 가장 큰 Contour를 선택해 `cv2.minAreaRect()`로 기울어진 사각형 생성.  
  - 사각형의 폭·높이를 정렬: \( \text{width} \geq \text{height} \).  
- 출력: \( (\text{width\_px}, \text{height\_px}) \) (픽셀 단위).

---

### 2. `pinhole_distance()`
- 입력:  
  - 물체의 픽셀 크기, 실제 크기, 카메라 초점 거리, 센서 크기, 이미지 해상도.  
- 처리:  
  - **핀홀 카메라 공식**으로 물체까지의 거리 추정:  
    \[
    Z = \frac{f \times \text{실제 크기} \times \text{이미지 해상도}}
             {\text{픽셀 크기} \times \text{센서 크기}}
    \]
- 출력: 물체까지 거리 (\(Z\), mm 단위).

---

### **핀홀 카메라 모델(Pinhole Camera Model)**

**공식**
   - 물체의 실제 크기(\(D_\text{real}\)), 물체가 투영된 픽셀 크기(\(D_\text{pixel}\)), 카메라 초점 거리(\(f\)), 그리고 센서 크기(\(S_\text{sensor}\)) 사이의 관계는 아래와 같다:
<div align="center">
  \[
  Z = \frac{f \cdot D_\text{real} \cdot I_\text{size}}{D_\text{pixel} \cdot S_\text{sensor}}
  \]
</div>

     
     - \( Z \): 물체까지의 거리(깊이).
     - \( f \): 카메라의 초점 거리(센서 기준, mm 단위).
     - \( D_\text{real} \): 물체의 실제 크기(현실 세계에서, mm 단위).
     - \( D_\text{pixel} \): 이미지에서 측정된 물체의 크기(픽셀 단위).
     - \( S_\text{sensor} \): 카메라 센서 크기(가로 또는 세로, mm 단위).
     - \( I_\text{size} \): 이미지의 해상도(가로 또는 세로, 픽셀 단위).

---

## 한계와 고려사항

1. **3D 기울어짐 보정의 한계**
   - 단일 이미지는 2D 투영 결과만 제공하므로, 3D 공간에서의 실제 크기와 차이가 있을 수 있음.  

2. **Segmentation 품질 의존**
   - 세그멘테이션이 정확하지 않으면 크기·깊이 계산의 정확도도 떨어짐.

3. **MiDaS 깊이 추정 오차**
   - MiDaS 모델 자체의 한계로 인해 반사, 투명체, 복잡한 배경에서 깊이 추정 오차 발생 가능.
