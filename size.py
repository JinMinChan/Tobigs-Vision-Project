import cv2
import json
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from pycocotools.mask import decode as decode_mask

# 하드코딩된 경로
IMAGE_PATH = "./test_image/size.jpg"        # 이미지 경로
RESULT_JSON_PATH = "./result/results.json"  # Segmentation 정보 경로

# 모니터의 실제 크기 (단위: mm) - 참조 물체
MONITOR_WIDTH_MM = 615.0
MONITOR_HEIGHT_MM = 365.0

# 카메라 내부 파라미터 (예: 갤럭시 S23 추정)
FOCAL_LENGTH_MM = 3.29    # 센서 기준 초점 거리 f
SENSOR_WIDTH_MM = 5.76   # 센서 가로 크기
SENSOR_HEIGHT_MM = 4.29  # 센서 세로 크기

def generate_relative_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    """
    MiDaS(DPT) 모델을 이용해 전체 이미지의 상대 깊이맵을 생성합니다.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    # MiDaS 결과를 원본 해상도에 맞게 리사이즈
    h, w, _ = image_bgr.shape
    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map_resized

def get_mask(segmentation):
    """
    COCO-style RLE segmentation을 디코딩하여 마스크(이진 배열) 반환.
    """
    mask = decode_mask(segmentation)  # 0/1 이진 mask
    return mask

def get_average_relative_depth(relative_depth_map, mask):
    """
    주어진 마스크 영역 내에서, MiDaS 상대 깊이의 평균값을 구한다.
    """
    ys, xs = np.where(mask > 0)
    values = relative_depth_map[ys, xs]
    return np.mean(values)

def measure_2d_size_from_mask(mask):
    """
    마스크 전체에서 contour(윤곽)을 찾고,
    minAreaRect (기울어진 사각형)으로부터 가장 긴 변(width), 짧은 변(height)을 픽셀 단위로 구한다.

    - mask: 0/1 (또는 0/255) 이진 이미지
    - return: (width_px, height_px)
      (width >= height)
    """
    # OpenCV 컨투어 찾기 (cv2.RETR_EXTERNAL: 가장 바깥 윤곽만, cv2.CHAIN_APPROX_SIMPLE: 윤곽점 단순화)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contour found in mask")

    # 가장 큰 컨투어(면적 기준) 선택 (모니터나 텀블러가 여러 조각이 아니라면, 보통 1개가 나오겠지만 혹시나 대비)
    max_contour = max(contours, key=cv2.contourArea)

    # 기울어진 사각형 구하기
    # minAreaRect() → ( (cx,cy), (w_rot,h_rot), angle ) 형태 반환
    rect = cv2.minAreaRect(max_contour)
    (w_rot, h_rot) = rect[1]  # 회전사각형의 폭, 높이

    # w_rot, h_rot 가 (0,0)이거나, 컨투어가 너무 작으면 예외처리
    if w_rot < 1e-5 or h_rot < 1e-5:
        raise ValueError("Invalid minAreaRect dimension")

    # width >= height 로 정렬
    width_px = max(w_rot, h_rot)
    height_px = min(w_rot, h_rot)

    return width_px, height_px

def pinhole_distance(
    object_pixel_size,  # BBox or RotatedRect에서 구한 픽셀 폭(또는 높이)
    object_real_size,   # 실제 폭(또는 높이, mm 단위)
    focal_length_mm,    # f (mm)
    sensor_size_mm,     # 센서 가로/세로 크기 (mm)
    image_size_px       # 이미지 가로/세로 해상도 (pixel)
):
    """
    핀홀 카메라 모델을 이용해,
    (픽셀 크기) + (물체 실제 크기) → 물체까지의 거리(Z, mm)를 추정하는 함수.
    
    Z = ( f * object_real_size * image_size_px ) / ( object_pixel_size * sensor_size_mm )
    """
    Z = ( focal_length_mm * object_real_size * image_size_px ) / ( object_pixel_size * sensor_size_mm )
    return Z

def main():
    # MiDaS 모델 로드
    midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    midas_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    midas_model.eval()

    # 이미지 로드
    original_bgr = cv2.imread(IMAGE_PATH)
    if original_bgr is None:
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {IMAGE_PATH}")
        return

    h_img, w_img, _ = original_bgr.shape

    # (1) MiDaS 상대 깊이맵 생성
    relative_depth_map = generate_relative_depth_map(original_bgr, midas_model, midas_processor)

    # (2) JSON 파일 로드 & Annotation
    with open(RESULT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data.get("annotations", [])

    #------------------------------------------------------
    # [A] 참조 물체(모니터) : 거리 & 스케일 구하기
    #------------------------------------------------------
    monitor_ann = next((ann for ann in annotations if ann["class_name"].lower() == "monitor"), None)
    if monitor_ann is None:
        print("[ERROR] 모니터 Segmentation 정보를 찾을 수 없습니다.")
        return

    monitor_mask = get_mask(monitor_ann["segmentation"])

    # 1) 모니터의 2D 픽셀 폭/높이 (RotatedRect 사용)
    monitor_px_w, monitor_px_h = measure_2d_size_from_mask(monitor_mask)

    # 2) 모니터 세그멘테이션 영역 내 평균 '상대 깊이'
    monitor_avg_rel_depth = get_average_relative_depth(relative_depth_map, monitor_mask)

    # 3) "가로 폭" 기준으로 모니터 거리 Z_monitor 추정 (정면 가정)
    #    원한다면 세로 폭도 추정해서 평균 or 더 정교한 계산 가능
    Z_monitor_estimated = pinhole_distance(
        object_pixel_size = monitor_px_w,
        object_real_size  = MONITOR_WIDTH_MM,
        focal_length_mm   = FOCAL_LENGTH_MM,
        sensor_size_mm    = SENSOR_WIDTH_MM,
        image_size_px     = w_img
    )

    # 4) 전역 스케일
    scale = Z_monitor_estimated / monitor_avg_rel_depth

    # 5) '절대 깊이 맵' 계산
    real_depth_map = relative_depth_map * scale

    print("=== Monitor Info ===")
    print(f"[INFO] monitor 2D size(px): width={monitor_px_w:.1f}, height={monitor_px_h:.1f}")
    print(f"[INFO] monitor avg rel depth: {monitor_avg_rel_depth:.4f}")
    print(f"[INFO] Z_monitor_estimated (mm): {Z_monitor_estimated:.2f}")
    print(f"[INFO] scale : {scale:.4f}")

    #------------------------------------------------------
    # [B] 텀블러(or 다른 객체) 실제 크기 계산
    #------------------------------------------------------
    tumbler_ann = next((ann for ann in annotations if ann["class_name"].lower() == "tumbler"), None)
    if tumbler_ann is None:
        print("[ERROR] 텀블러 Segmentation 정보를 찾을 수 없습니다.")
        return

    tumbler_mask = get_mask(tumbler_ann["segmentation"])
    tumbler_px_w, tumbler_px_h = measure_2d_size_from_mask(tumbler_mask)

    # 텀블러 영역의 평균 '절대 깊이'
    tumbler_avg_depth = get_average_relative_depth(real_depth_map, tumbler_mask)

    # 핀홀 공식으로 텀블러 가로/세로 크기 계산
    # 가로 폭 계산 (이미지 가로 기준)
    tumbler_real_width = (
        tumbler_px_w * tumbler_avg_depth * SENSOR_WIDTH_MM
    ) / (FOCAL_LENGTH_MM * w_img)

    # 세로 높이 계산 (이미지 세로 기준)
    tumbler_real_height = (
        tumbler_px_h * tumbler_avg_depth * SENSOR_HEIGHT_MM
    ) / (FOCAL_LENGTH_MM * h_img)

    print("\n=== Tumbler Info ===")
    print(f"[INFO] tumbler 2D size(px): width={tumbler_px_w:.1f}, height={tumbler_px_h:.1f}")
    print(f"[INFO] tumbler avg depth (mm): {tumbler_avg_depth:.2f}")
    print(f"[RESULT] Tumbler Width (mm) : {tumbler_real_width:.2f}")
    print(f"[RESULT] Tumbler Height (mm): {tumbler_real_height:.2f}")

if __name__ == "__main__":
    main()

