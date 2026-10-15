import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


# =========================================================
# Phase 모델 설정
# =========================================================

NUM_PHASE_CLASSES = 4

PATCH_SIZE = 512
PATCH_STRIDE = 256

# 학습할 때 사용한 클래스 순서와 반드시 동일해야 함
PHASE_CLASS_NAMES = [
    "Al",
    "Primary Si",
    "Al3Ni",
    "Eutectic Si",
]

# Phase 시각화 색상 (RGB)
PHASE_CLASS_COLORS = {
    0: (153, 127, 76),    # Al
    1: (76, 178, 76),     # Primary Si
    2: (25, 76, 153),     # Al3Ni
    3: (204, 204, 204),   # Eutectic Si
}


# ImageNet normalization
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# =========================================================
# 모델 생성
# =========================================================

def create_phase_model():
    """
    학습에 사용한 모델 구조:
    MiT-B0 encoder + U-Net decoder
    """

    model = smp.Unet(
        encoder_name="mit_b0",

        # 전체 checkpoint를 불러오기 때문에
        # 서버에서 ImageNet weight를 다시 다운로드하지 않음
        encoder_weights=None,

        in_channels=3,
        classes=NUM_PHASE_CLASSES,

        decoder_channels=(
            256,
            128,
            64,
            32,
            16,
        ),
    )

    return model


# =========================================================
# 체크포인트 로드
# =========================================================

def load_phase_model(model_path, device):
    print(f"Phase 모델 로드 중: {model_path}")

    model = create_phase_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    # 학습 checkpoint가
    # {"model_state_dict": ...} 형태인 경우
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    # state_dict 자체를 저장한 경우도 대응
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Phase 모델 로드 완료")

    return model


# =========================================================
# 이미지 전처리
# =========================================================

def preprocess_patch(patch):
    """
    Albumentations 없이 학습 당시 ImageNet normalization을
    직접 적용한다.

    입력:
        RGB numpy array
        shape = (H, W, 3)

    출력:
        torch.Tensor
        shape = (3, H, W)
    """

    patch = patch.astype(np.float32) / 255.0

    patch = (
        patch - IMAGENET_MEAN
    ) / IMAGENET_STD

    # HWC -> CHW
    patch = np.transpose(
        patch,
        (2, 0, 1),
    )

    # 메모리 연속성 보장
    patch = np.ascontiguousarray(patch)

    tensor = torch.from_numpy(patch).float()

    return tensor


# =========================================================
# Sliding Window 위치 계산
# =========================================================

def _get_patch_positions(length, patch_size, stride):
    """
    이미지 끝부분까지 빠짐없이 추론하기 위한
    sliding-window 시작 위치를 계산한다.
    """

    if length <= patch_size:
        return [0]

    positions = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    last_position = length - patch_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


# =========================================================
# Phase Segmentation 추론
# =========================================================

@torch.no_grad()
def predict_phase(model, image_rgb, device):
    """
    512x512 sliding-window 방식으로 추론한다.

    겹치는 영역에서는 softmax probability를 평균낸 뒤
    최종 argmax를 수행한다.

    입력:
        image_rgb:
            RGB numpy array (H, W, 3)

    반환:
        mask:
            클래스 index mask (H, W)
    """

    original_h, original_w = image_rgb.shape[:2]

    image = image_rgb.copy()

    # -----------------------------------------------------
    # 이미지가 512보다 작은 경우 padding
    # -----------------------------------------------------

    pad_h = max(0, PATCH_SIZE - original_h)
    pad_w = max(0, PATCH_SIZE - original_w)

    if pad_h > 0 or pad_w > 0:

        # reflect padding이 불가능한 극단적으로 작은 이미지에
        # 대비하여 우선 reflect를 사용하고 실패하면 edge 사용
        try:
            image = np.pad(
                image,
                (
                    (0, pad_h),
                    (0, pad_w),
                    (0, 0),
                ),
                mode="reflect",
            )

        except ValueError:
            image = np.pad(
                image,
                (
                    (0, pad_h),
                    (0, pad_w),
                    (0, 0),
                ),
                mode="edge",
            )

    h, w = image.shape[:2]

    y_positions = _get_patch_positions(
        h,
        PATCH_SIZE,
        PATCH_STRIDE,
    )

    x_positions = _get_patch_positions(
        w,
        PATCH_SIZE,
        PATCH_STRIDE,
    )

    # 클래스별 확률 누적
    probability_sum = np.zeros(
        (
            NUM_PHASE_CLASSES,
            h,
            w,
        ),
        dtype=np.float32,
    )

    # 각 픽셀이 몇 번 예측되었는지
    count_map = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Sliding Window 추론
    # -----------------------------------------------------

    for y in y_positions:

        for x in x_positions:

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ]

            tensor = preprocess_patch(patch)

            tensor = tensor.unsqueeze(0).to(device)

            logits = model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            probabilities = (
                probabilities
                .squeeze(0)
                .cpu()
                .numpy()
            )

            probability_sum[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ] += probabilities

            count_map[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ] += 1.0

    # 0 division 방지
    count_map = np.maximum(
        count_map,
        1.0,
    )

    # 겹친 patch의 확률 평균
    probability_sum /= count_map[None, :, :]

    # 최종 클래스 결정
    mask = np.argmax(
        probability_sum,
        axis=0,
    ).astype(np.uint8)

    # padding 제거
    mask = mask[
        :original_h,
        :original_w,
    ]

    return mask


# =========================================================
# Mask -> RGB 이미지
# =========================================================

def phase_mask_to_rgb(mask):
    """
    클래스 index mask를 RGB 컬러 mask로 변환한다.
    """

    h, w = mask.shape

    color_mask = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for class_id, color in PHASE_CLASS_COLORS.items():

        color_mask[
            mask == class_id
        ] = color

    return color_mask


# =========================================================
# Phase 비율 계산
# =========================================================

def calculate_phase_distribution(mask):
    """
    각 Phase가 이미지에서 차지하는 pixel 비율을 계산한다.
    """

    total_pixels = mask.size

    distribution = {}

    for class_id, class_name in enumerate(
        PHASE_CLASS_NAMES
    ):

        pixel_count = int(
            np.sum(mask == class_id)
        )

        percentage = (
            pixel_count / total_pixels
        ) * 100.0

        distribution[class_name] = round(
            percentage,
            2,
        )

    return distribution


# =========================================================
# 원본 + Segmentation Overlay
# =========================================================

def create_phase_overlay(
    image_rgb,
    mask,
    alpha=0.45,
):
    """
    원본 이미지 위에 Phase segmentation 결과를
    반투명하게 합성한다.
    """

    color_mask = phase_mask_to_rgb(mask)

    overlay = cv2.addWeighted(
        image_rgb,
        1.0 - alpha,
        color_mask,
        alpha,
        0,
    )

    return overlay