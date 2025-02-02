import cv2
import numpy as np
import argparse
import os

# 명령행 인자 설정
parser = argparse.ArgumentParser(description='Apply LOG filter to an image')
parser.add_argument('--work_dir', type=str, required=True, help='작업 디렉토리 경로')
args = parser.parse_args()

# 작업 디렉토리 설정
work_dir = args.work_dir
os.makedirs(work_dir, exist_ok=True)

# 이미지 불러오기
image_path = os.path.join(work_dir, 'images','sample1_6x9.jpg')
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Gaussian 블러 적용
blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

# Laplacian 필터 적용
log_result = cv2.Laplacian(blurred, cv2.CV_64F)

# 결과 저장
output_path = os.path.join(work_dir, 'log_output.jpg')
cv2.imwrite(output_path, log_result)