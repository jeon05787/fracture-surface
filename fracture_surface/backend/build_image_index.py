#이미지DB에서 이미지를 읽어와서 모델로 특징 벡터를 추출한 후
#벡터와 이미지 경로를 각각 .npy와 .json 파일로 저장하는 스크립트
#image_index폴더가 없거나, image_db폴더에 변경이 생길 시 실행하여 인덱스를 갱신해야 함
#image_db폴더는 공용 드라이브에 업로드됨, backend폴더에 넣고 실행
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model import FractographyNet
from torchvision import transforms


# =========================
# 경로 설정
# =========================
IMAGE_DB_DIR = "image_db"
INDEX_DIR = "image_index"

FEATURE_SAVE_PATH = os.path.join(INDEX_DIR, "image_features.npy")
PATH_SAVE_PATH = os.path.join(INDEX_DIR, "image_paths.json")

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 전처리
# =========================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================
# 이미지 경로 수집
# =========================
def get_image_paths(root_dir):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    image_paths = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)

                # backend 기준 상대경로로 저장
                relative_path = os.path.relpath(full_path, start=".")
                relative_path = relative_path.replace("\\", "/")

                image_paths.append(relative_path)

    return image_paths


# =========================
# 특징 벡터 추출
# =========================
def extract_feature(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x = image_tensor

        # ConvNeXt backbone
        x = model.backbone(x)

        # ASPP
        x = model.aspp(x)

        # 만약 모델에 CBAM attention이 있으면 사용
        if hasattr(model, "attention"):
            x = model.attention(x)

        # GAP + Flatten
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # cosine similarity용 정규화
        x = F.normalize(x, dim=1)

    return x.cpu().numpy()[0]


# =========================
# 메인
# =========================
def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"디바이스: {DEVICE}")
    print("모델 생성 중...")

    model = FractographyNet(num_classes=4).to(DEVICE)
    model.eval()

    print("이미지 경로 수집 중...")
    image_paths = get_image_paths(IMAGE_DB_DIR)

    if len(image_paths) == 0:
        raise RuntimeError(f"{IMAGE_DB_DIR} 안에서 이미지를 찾지 못했습니다.")

    print(f"총 이미지 수: {len(image_paths)}장")

    features = []

    for path in tqdm(image_paths, desc="특징 벡터 추출 중"):
        try:
            feature = extract_feature(model, path)
            features.append(feature)
        except Exception as e:
            print(f"[경고] 처리 실패: {path} / {e}")

    features = np.array(features, dtype=np.float32)

    np.save(FEATURE_SAVE_PATH, features)

    with open(PATH_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)

    print("\n완료!")
    print(f"특징 벡터 저장: {FEATURE_SAVE_PATH}")
    print(f"이미지 경로 저장: {PATH_SAVE_PATH}")
    print(f"벡터 shape: {features.shape}")


if __name__ == "__main__":
    main()
