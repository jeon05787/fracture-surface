# ═══════════════════════════════════════════════════════
# 기본 라이브러리 import
# ═══════════════════════════════════════════════════════

import base64
import io
import os

import cv2
import numpy as np
import torch

from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from torchvision import transforms


# ═══════════════════════════════════════════════════════
# 프로젝트 내부 모듈 import
# ═══════════════════════════════════════════════════════

# 유사 이미지 검색
from similar_search import SimilarImageSearcher

# Phase Segmentation
from phase_model import (
    load_phase_model,
    predict_phase,
    phase_mask_to_rgb,
    calculate_phase_distribution,
    create_phase_overlay,
)

# CNN 모델 로드
from model import load_model

# Grad-CAM++
from gradcam import (
    GradCAMPlusPlus,
    cam_to_mask,
    split_solo_overlap,
    build_contour_image,
    draw_dashed_contour,
    extract_contours_json,
    to_b64,
)

# LLM 분석
from llm_service import (
    generate_llm_analysis,
    generate_compare_analysis,
)


# ═══════════════════════════════════════════════════════
# FastAPI 앱 생성
# ═══════════════════════════════════════════════════════

app = FastAPI(
    title="Fractography Analysis API"
)


# ═══════════════════════════════════════════════════════
# 유사 이미지 폴더 설정
# ═══════════════════════════════════════════════════════

SIMILAR_DIR = os.path.join(
    os.path.dirname(__file__),
    "similar_db",
)

# 폴더가 없으면 자동 생성
os.makedirs(
    SIMILAR_DIR,
    exist_ok=True,
)

for class_name in [
    "Cleavage",
    "Ductile",
    "Fatigue",
    "Intergranular",
]:
    os.makedirs(
        os.path.join(
            SIMILAR_DIR,
            class_name,
        ),
        exist_ok=True,
    )


# 프론트에서 유사 이미지 접근 가능
app.mount(
    "/similar_db",
    StaticFiles(
        directory=SIMILAR_DIR
    ),
    name="similar_db",
)


# ═══════════════════════════════════════════════════════
# CORS 설정
# ═══════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# 디바이스 / 모델 / 클래스 설정
# ═══════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fractography_best5.pth",
)


IMG_SIZE = 224
NUM_CLASSES = 4


CNN_CLASSES = [
    "Cleavage",
    "Ductile",
    "Fatigue",
    "Intergranular",
]


KO_LABELS = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}

# ═══════════════════════════════════════════════════════
# Grad-CAM 색상
# BGR 기준
# ═══════════════════════════════════════════════════════

CLASS_COLORS_BGR = {
    "Cleavage": (
        245,
        130,
        59,
    ),

    "Ductile": (
        94,
        197,
        34,
    ),

    "Fatigue": (
        21,
        204,
        250,
    ),

    "Intergranular": (
        68,
        68,
        239,
    ),
}


# ═══════════════════════════════════════════════════════
# CNN 모델 로드
# ═══════════════════════════════════════════════════════

model = load_model(
    model_path=MODEL_PATH,
    device=DEVICE,
    num_classes=NUM_CLASSES,
)

# ═══════════════════════════════════════════════════════
# Phase Segmentation 모델 로드
# ═══════════════════════════════════════════════════════

PHASE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "phase_best.pth",
)

phase_model = load_phase_model(
    model_path=PHASE_MODEL_PATH,
    device=DEVICE,
)


# ═══════════════════════════════════════════════════════
# 유사 이미지 검색기
# ═══════════════════════════════════════════════════════

similar_searcher = SimilarImageSearcher(
    model=model,
    device=DEVICE,
    class_names=CNN_CLASSES,
    similar_dir=SIMILAR_DIR,
    cache_path=os.path.join(
        os.path.dirname(__file__),
        "similar_cache.pt",
    ),
    image_size=IMG_SIZE,
)


# ═══════════════════════════════════════════════════════
# Grad-CAM++
# ═══════════════════════════════════════════════════════

gradcam_all = GradCAMPlusPlus(
    model
)


# ═══════════════════════════════════════════════════════
# 이미지 전처리
# ═══════════════════════════════════════════════════════

preprocess = transforms.Compose([
    transforms.Resize(
        IMG_SIZE + 32
    ),

    transforms.CenterCrop(
        IMG_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [
            0.485,
            0.456,
            0.406,
        ],
        [
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ═══════════════════════════════════════════════════════
# 컬러 RGBA 레이어 생성
# gradcam_layers 폴백용
# ═══════════════════════════════════════════════════════

def build_layer_rgba(
    img_rgb,
    name,
    solo_mask,
    overlap_mask,
):

    H, W = img_rgb.shape[:2]

    bgr_c = CLASS_COLORS_BGR[
        name
    ]

    canvas = np.zeros(
        (
            H,
            W,
            4,
        ),
        dtype=np.uint8,
    )


    # ─────────────────────────────────────────────
    # 단독 영역 → 실선
    # ─────────────────────────────────────────────

    if (
        solo_mask is not None
        and solo_mask.sum() > 0
    ):

        tmp_solo = np.zeros(
            (
                H,
                W,
                3,
            ),
            dtype=np.uint8,
        )

        cnts, _ = cv2.findContours(
            solo_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in cnts:

            cv2.drawContours(
                tmp_solo,
                [cnt],
                -1,
                bgr_c,
                thickness=3,
                lineType=cv2.LINE_AA,
            )


        solo_pixel = np.any(
            tmp_solo > 0,
            axis=2,
        )

        canvas[
            solo_pixel,
            0,
        ] = bgr_c[2]

        canvas[
            solo_pixel,
            1,
        ] = bgr_c[1]

        canvas[
            solo_pixel,
            2,
        ] = bgr_c[0]

        canvas[
            solo_pixel,
            3,
        ] = 255


    # ─────────────────────────────────────────────
    # 겹침 영역 → 점선
    # ─────────────────────────────────────────────

    if (
        overlap_mask is not None
        and overlap_mask.sum() > 0
    ):

        tmp_overlap = np.zeros(
            (
                H,
                W,
                3,
            ),
            dtype=np.uint8,
        )

        cnts, _ = cv2.findContours(
            overlap_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in cnts:

            draw_dashed_contour(
                tmp_overlap,
                cnt,
                bgr_c,
                thickness=3,
                dash_length=12,
            )


        overlap_pixel = np.any(
            tmp_overlap > 0,
            axis=2,
        )

        canvas[
            overlap_pixel,
            0,
        ] = bgr_c[2]

        canvas[
            overlap_pixel,
            1,
        ] = bgr_c[1]

        canvas[
            overlap_pixel,
            2,
        ] = bgr_c[0]

        canvas[
            overlap_pixel,
            3,
        ] = 255


    return canvas


# ═══════════════════════════════════════════════════════
# 이진 마스크 → PNG base64
# 동적 Grad-CAM 시각화용
# ═══════════════════════════════════════════════════════

def mask_to_b64_png(
    mask: np.ndarray
) -> str:

    pil = Image.fromarray(
        mask.astype(
            np.uint8
        ),
        mode="L",
    )

    buf = io.BytesIO()

    pil.save(
        buf,
        format="PNG",
        optimize=True,
    )

    b64 = base64.b64encode(
        buf.getvalue()
    ).decode()

    return (
        f"data:image/png;base64,{b64}"
    )


# ═══════════════════════════════════════════════════════
# 서버 상태 확인
# ═══════════════════════════════════════════════════════

@app.get("/")
async def root():

    return {
        "message":
            "Fractography API is running",

        "model":
            "ConvNeXt-Small + ASPP",

        "pth":
            MODEL_PATH,

        "device":
            str(
                DEVICE
            ),
    }


# ═══════════════════════════════════════════════════════
# 메인 분석 API
# ═══════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze_fracture(
    file: UploadFile = File(...),
    material: str = Form(""),
    conf_thresh: float = Form(0.05),
    cam_percentile: float = Form(80.0),
    min_area_ratio: float = Form(0.005),
):

    print(
        f"요청 수신 — 재질: {material}"
    )

    model.eval()


    # ═══════════════════════════════════════════════════
    # 이미지 읽기
    # ═══════════════════════════════════════════════════

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(
            image_bytes
        )
    ).convert(
        "RGB"
    )

    img_rgb = np.array(
        image
    )

    H, W = img_rgb.shape[:2]


    # ═══════════════════════════════════════════════════
    # 이미지 전처리
    # ═══════════════════════════════════════════════════

    input_tensor = (
        preprocess(
            image
        )
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    # ═══════════════════════════════════════════════════
    # CNN 추론
    # ═══════════════════════════════════════════════════

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probs_tensor = torch.softmax(
            output,
            dim=1,
        )[0]


    sorted_probs, sorted_indices = torch.sort(
        probs_tensor,
        descending=True,
    )


    top1_idx = (
        sorted_indices[
            0
        ].item()
    )

    top2_idx = (
        sorted_indices[
            1
        ].item()
    )


    top1_percent = (
        sorted_probs[
            0
        ].item()
        * 100
    )

    top2_percent = (
        sorted_probs[
            1
        ].item()
        * 100
    )


    gap = (
        top1_percent
        - top2_percent
    )


    top1_en = CNN_CLASSES[
        top1_idx
    ]

    top2_en = CNN_CLASSES[
        top2_idx
    ]


    top1_label = KO_LABELS[
        top1_en
    ]

    top2_label = KO_LABELS[
        top2_en
    ]


    # ═══════════════════════════════════════════════════
    # Grad-CAM++
    # ═══════════════════════════════════════════════════

    gradcam_image = None

    gradcam_layers = {}

    gradcam_masks = {}

    base_image = None

    gradcam_contours = {}

    masks_dict = {}


    try:

        cams_dict, probs_np = (
            gradcam_all
            .generate_all_classes(
                input_tensor,
                num_classes=NUM_CLASSES,
            )
        )


        # ─────────────────────────────────────────────
        # 클래스별 Grad-CAM 마스크 생성
        # ─────────────────────────────────────────────

        for i, name in enumerate(
            CNN_CLASSES
        ):

            if (
                probs_np[i]
                < conf_thresh
            ):
                continue


            mask = cam_to_mask(
                cams_dict[i],
                (
                    W,
                    H,
                ),
                cam_percentile=cam_percentile,
                min_area_ratio=min_area_ratio,
            )


            if mask.sum() > 0:

                masks_dict[
                    name
                ] = mask


        # ─────────────────────────────────────────────
        # 단독 / 겹침 영역 분리
        # ─────────────────────────────────────────────

        if masks_dict:

            (
                solo_masks,
                overlap_masks,
            ) = split_solo_overlap(
                masks_dict
            )

        else:

            solo_masks = {}
            overlap_masks = {}


        # ─────────────────────────────────────────────
        # 통합 Grad-CAM 이미지
        # ─────────────────────────────────────────────

        gradcam_image = to_b64(
            build_contour_image(
                img_rgb,
                solo_masks,
                overlap_masks,
            )
        )


        # ─────────────────────────────────────────────
        # 클래스별 컬러 레이어
        # ─────────────────────────────────────────────

        for name in CNN_CLASSES:

            s = solo_masks.get(
                name,
                np.zeros(
                    (
                        H,
                        W,
                    ),
                    dtype=np.uint8,
                ),
            )

            o = overlap_masks.get(
                name,
                np.zeros(
                    (
                        H,
                        W,
                    ),
                    dtype=np.uint8,
                ),
            )


            gradcam_layers[
                name
            ] = to_b64(
                build_layer_rgba(
                    img_rgb,
                    name,
                    s,
                    o,
                )
            )


        # ─────────────────────────────────────────────
        # 동적 시각화용 흑백 마스크
        # ─────────────────────────────────────────────

        for name in CNN_CLASSES:

            mask = masks_dict.get(
                name,
                np.zeros(
                    (
                        H,
                        W,
                    ),
                    dtype=np.uint8,
                ),
            )

            gradcam_masks[
                name
            ] = mask_to_b64_png(
                mask
            )


        base_image = to_b64(
            img_rgb
        )


        gradcam_contours = (
            extract_contours_json(
                masks_dict,
                (
                    H,
                    W,
                ),
            )
        )


    except Exception as e:

        print(
            f"Grad-CAM++ 레이어 생성 오류: {e}"
        )


    # ═══════════════════════════════════════════════════
    # 최종 파손 유형 결정
    # ═══════════════════════════════════════════════════

    ductile_prob = probs_tensor[
        CNN_CLASSES.index(
            "Ductile"
        )
    ].item()


    # ═══════════════════════════════════════════════════
    # 최종 파손 유형 결정
    # Fatigue가 검출되고 확률 60% 이상이면 Fatigue 우선
    # ═══════════════════════════════════════════════════

    FATIGUE_THRESHOLD = 0.60

    detected = [name for name in CNN_CLASSES if name in masks_dict]

    fatigue_prob = probs_tensor[CNN_CLASSES.index("Fatigue")].item()

    if "Fatigue" in detected and fatigue_prob >= FATIGUE_THRESHOLD:
        final_en = "Fatigue"
        decision_path = (
            f"Fatigue 검출 및 확률 {fatigue_prob * 100:.1f}% "
            f"(≥{FATIGUE_THRESHOLD * 100:.0f}%) → Fatigue 우선"
        )

    elif detected:
        final_en = max(
            detected,
            key=lambda n: probs_tensor[CNN_CLASSES.index(n)].item(),
        )
        decision_path = f"검출 클래스 중 최고 확률 ({final_en})"

    else:
        final_en = CNN_CLASSES[int(probs_tensor.argmax().item())]
        decision_path = "폴백: 검출 클래스 없음 → argmax"
        print("[경고] GradCAM 검출 클래스 없음 → softmax argmax 폴백")

    # ═══════════════════════════════════════════════════
    # 최종 분석 정보
    # ═══════════════════════════════════════════════════

    final_label = KO_LABELS[
        final_en
    ]

    final_idx = CNN_CLASSES.index(
        final_en
    )

    final_percent = (
        probs_tensor[
            final_idx
        ].item()
        * 100
    )


    print(
        f"[결정 경로] {decision_path}"
    )


    is_mixed = False

    highlighted_types = [
        final_label
    ]

    display_prediction = (
        final_label
    )

    pred_en = (
        final_en
    )

    prediction = (
        final_label
    )

    confidence = (
        f"{final_percent:.1f}%"
    )


    # ═══════════════════════════════════════════════════
    # 유사 이미지 검색
    # ═══════════════════════════════════════════════════

    try:

        similar_images = (
            similar_searcher
            .find_similar_images(
                image=image,
                predicted_class=pred_en,
                top_k=3,
            )
        )


    except Exception as e:

        print(
            f"유사 이미지 검색 오류: {e}"
        )

        similar_images = []


    # ═══════════════════════════════════════════════════
    # 신뢰도 상태
    # ═══════════════════════════════════════════════════

    if final_percent >= 80:

        confidence_status = (
            "high"
        )

        confidence_message = (
            "현재 분석은 신뢰할 수 있는 결과입니다."
        )


    elif final_percent >= 60:

        confidence_status = (
            "medium"
        )

        confidence_message = (
            "결과 해석에 주의가 필요합니다."
        )


    else:

        confidence_status = (
            "low"
        )

        confidence_message = (
            "예측확률이 낮아 오분류 가능성이 있습니다. "
            "추가 이미지나 전문가 검토가 필요할 수 있습니다."
        )


    # ═══════════════════════════════════════════════════
    # 클래스별 확률
    # ═══════════════════════════════════════════════════

    # Grad-CAM 자체가 실패해 masks_dict가 비어 있으면 전부 표시
    show_all = len(masks_dict) == 0

    similarities = {}
    for i, name in enumerate(CNN_CLASSES):
        if show_all or name in masks_dict or name == final_en:
            similarities[KO_LABELS[name]] = f"{probs_tensor[i].item() * 100:.1f}%"
        else:
            similarities[KO_LABELS[name]] = "-"


    # ═══════════════════════════════════════════════════
    # Gemma 멀티모달 설명 생성
    # ═══════════════════════════════════════════════════

    llm_result = (
        generate_llm_analysis(
            prediction=prediction,
            confidence_percent=final_percent,
            material=material,
            image_bytes=image_bytes,
        )
    )


    # ═══════════════════════════════════════════════════
    # 최종 응답
    # ═══════════════════════════════════════════════════

    return {

        # ─────────────────────────────────────────────
        # CNN 결과
        # ─────────────────────────────────────────────

        "prediction":
            prediction,

        "prediction_en":
            pred_en,

        "display_prediction":
            display_prediction,

        "confidence":
            confidence,

        "similarities":
            similarities,

        "is_mixed":
            is_mixed,

        "mixed_gap":
            f"{gap:.1f}%",

        "top1_type":
            top1_label,

        "top2_type":
            top2_label,

        "highlighted_types":
            highlighted_types,


        # ─────────────────────────────────────────────
        # LLM 이미지 분석
        # ─────────────────────────────────────────────

        "feature":
            llm_result[
                "feature"
            ],

        "visual_observation_1":
            llm_result.get(
                "visual_observation_1",
                "",
            ),

        "visual_observation_2":
            llm_result.get(
                "visual_observation_2",
                "",
            ),

        "mechanism_relation":
            llm_result.get(
                "mechanism_relation",
                "",
            ),

        "cause":
            llm_result.get(
                "cause",
                llm_result.get(
                    "expected_cause",
                    "",
                ),
            ),

        "expected_cause":
            llm_result[
                "expected_cause"
            ],

        "cause_explanation":
            llm_result.get(
                "cause_explanation",
                "",
            ),

        "explanation":
            llm_result[
                "explanation"
            ],


        # 전체 LLM 구조도 같이 보존
        "llm_analysis":
            llm_result,


        # ─────────────────────────────────────────────
        # 입력 재질
        # ─────────────────────────────────────────────

        "material":
            material,


        # ─────────────────────────────────────────────
        # 신뢰도 상태
        # ─────────────────────────────────────────────

        "confidence_status":
            confidence_status,

        "confidence_message":
            confidence_message,


        # ─────────────────────────────────────────────
        # Grad-CAM
        # ─────────────────────────────────────────────

        "gradcam_image":
            gradcam_image,

        "gradcam_layers":
            gradcam_layers,

        "gradcam_masks":
            gradcam_masks,

        "base_image":
            base_image,

        "gradcam_contours":
            gradcam_contours,


        # ─────────────────────────────────────────────
        # 유사 이미지
        # ─────────────────────────────────────────────

        "similar_images":
            similar_images,
    }


# ═══════════════════════════════════════════════════════
# 분석 결과 비교 API
# ═══════════════════════════════════════════════════════

@app.post("/compare")
async def compare_analysis(
    payload: dict
):

    items = payload.get(
        "items",
        [],
    )


    # ═══════════════════════════════════════════════════
    # 최소 2개 필요
    # ═══════════════════════════════════════════════════

    if len(
        items
    ) < 2:

        return {
            "summary":
                "비교하려면 최소 2개의 분석 결과가 필요합니다.",

            "common_point":
                "",

            "visual_difference":
                "",

            "mechanism_difference":
                "",

            "cause_difference":
                "",

            "confidence_difference":
                "",

            "final_opinion":
                "두 개 이상의 분석 결과를 선택해 주세요.",

            # 기존 프론트 호환
            "compare_summary":
                "비교하려면 최소 2개의 분석 결과가 필요합니다.",
        }


    # ═══════════════════════════════════════════════════
    # Gemma 비교 분석
    # ═══════════════════════════════════════════════════

    try:

        compare_result = (
            generate_compare_analysis(
                items
            )
        )


    except Exception as e:

        print(
            f"비교 설명 생성 오류: {e}"
        )


        compare_result = {
            "summary":
                "선택한 두 분석 결과를 비교하는 과정에서 "
                "LLM 분석 오류가 발생했습니다.",

            "common_point":
                "두 결과의 공통적인 특징은 "
                "개별 분석 결과를 기준으로 확인할 수 있습니다.",

            "visual_difference":
                "두 이미지에서 관찰된 표면 형태와 "
                "분포를 함께 비교할 필요가 있습니다.",

            "mechanism_difference":
                "각 이미지에서 관찰된 특징과 "
                "CNN 파손 메커니즘의 관계를 비교할 수 있습니다.",

            "cause_difference":
                "예상 원인은 각 분석에서 제시된 "
                "가능성 범위 안에서 참고하는 것이 좋습니다.",

            "confidence_difference":
                "예측 확률을 함께 확인하여 "
                "결과의 신뢰도를 비교하는 것이 좋습니다.",

            "final_opinion":
                "두 결과를 함께 참고하되, "
                "이미지만으로 실제 파손 원인을 "
                "확정하기는 어렵습니다.",
        }


    # ═══════════════════════════════════════════════════
    # 혹시 문자열로 반환된 경우
    # ═══════════════════════════════════════════════════

    if isinstance(
        compare_result,
        str,
    ):

        return {
            "summary":
                compare_result,

            "common_point":
                "",

            "visual_difference":
                "",

            "mechanism_difference":
                "",

            "cause_difference":
                "",

            "confidence_difference":
                "",

            "final_opinion":
                compare_result,

            "compare_summary":
                compare_result,
        }


    # ═══════════════════════════════════════════════════
    # 기존 프론트 호환
    #
    # 핵심 요약 영역에는 final_opinion이 아니라
    # summary를 넣는 것이 자연스러움
    # ═══════════════════════════════════════════════════

    compare_result[
        "compare_summary"
    ] = compare_result.get(
        "summary",
        compare_result.get(
            "final_opinion",
            "분석 결과 비교가 완료되었습니다.",
        ),
    )


    return compare_result

# ═══════════════════════════════════════════════════════
# Phase Segmentation API
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# Phase Segmentation API
# ═══════════════════════════════════════════════════════

@app.post("/analyze-phase")
async def analyze_phase(
    file: UploadFile = File(...)
):

    print(
        f"Phase 분석 요청 수신: {file.filename}"
    )

    # ==========================================
    # 1. 이미지 파일 읽기
    # ==========================================

    image_bytes = await file.read()

    if not image_bytes:

        return {
            "status": "error",
            "message": "이미지 파일이 비어 있습니다.",
        }

    # ==========================================
    # 2. 이미지 형식 확인
    # ==========================================

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as e:

        print(
            f"Phase 이미지 로드 오류: {e}"
        )

        return {
            "status": "error",
            "message": "이미지를 읽을 수 없습니다.",
        }

    # PIL -> NumPy RGB
    image_rgb = np.array(image)

    height, width = image_rgb.shape[:2]

    print(
        f"Phase 이미지 크기: {width} x {height}"
    )

    # ==========================================
    # 3. Phase Segmentation 추론
    # ==========================================

    try:

        phase_model.eval()

        mask = predict_phase(
            model=phase_model,
            image_rgb=image_rgb,
            device=DEVICE,
        )

        print(
            f"Phase 추론 완료 - mask shape: {mask.shape}"
        )

    except Exception as e:

        print(
            f"Phase 모델 추론 오류: {e}"
        )

        return {
            "status": "error",
            "message": (
                "Phase Segmentation 모델 추론 중 "
                "오류가 발생했습니다."
            ),
        }

    # ==========================================
    # 4. Phase 면적 비율 계산
    # ==========================================

    try:

        phase_distribution = (
            calculate_phase_distribution(
                mask
            )
        )

        print(
            f"Phase 분포: {phase_distribution}"
        )

    except Exception as e:

        print(
            f"Phase 분포 계산 오류: {e}"
        )

        return {
            "status": "error",
            "message": (
                "Phase 분포 계산 중 "
                "오류가 발생했습니다."
            ),
        }

    # ==========================================
    # 5. 컬러 Mask 생성
    # ==========================================

    color_mask = phase_mask_to_rgb(
        mask
    )

    # ==========================================
    # 6. Overlay 생성
    # ==========================================

    overlay = create_phase_overlay(
        image_rgb=image_rgb,
        mask=mask,
        alpha=0.45,
    )

    # ==========================================
    # 7. NumPy 이미지 -> Base64 PNG
    # ==========================================

    try:

        mask_image = to_b64(
            color_mask
        )

        overlay_image = to_b64(
            overlay
        )

    except Exception as e:

        print(
            f"Phase 이미지 변환 오류: {e}"
        )

        return {
            "status": "error",
            "message": (
                "Phase 결과 이미지 생성 중 "
                "오류가 발생했습니다."
            ),
        }

    # ==========================================
    # 8. 최종 결과 반환
    # ==========================================

    return {
        "status": "success",

        "message":
            "Phase Segmentation 분석이 완료되었습니다.",

        "image_width":
            width,

        "image_height":
            height,

        "overlay_image":
            overlay_image,

        "mask_image":
            mask_image,

        "phase_distribution":
            phase_distribution,
    }

# ═══════════════════════════════════════════════════════
# 직접 실행
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )