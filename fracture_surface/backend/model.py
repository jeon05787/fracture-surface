import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_small


# ==============================
# 클래스 상수
# ==============================

CLASS_NAMES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]

KO_NAMES = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}

CLASS_COLORS_BGR = {
    "Cleavage": (245, 130, 59),
    "Ductile": (94, 197, 34),
    "Fatigue": (21, 204, 250),
    "Intergranular": (68, 68, 239),
}

CLASS_FEATURES = {
    "Cleavage": "평평하고 반짝이는 파단면, 결정면을 따라 직선적으로 쪼개진 형태",
    "Ductile": "딤플(dimple) 패턴, 컵-콘 형태의 소성 변형 흔적",
    "Fatigue": "비치 마크(beach mark), 줄무늬 형태의 균열 전파 흔적",
    "Intergranular": "결정립 경계가 드러난 표면, 입자 경계를 따라 전파된 균열",
}

CLASS_CAUSES = {
    "Cleavage": "충격 하중, 저온 환경, 응력 집중, 급격한 변형속도",
    "Ductile": "과도한 인장 하중, 정적 과부하",
    "Fatigue": "장기간 반복 하중, 진동, 응력 집중부 초기 균열",
    "Intergranular": "수소 취성, 응력 부식 균열, 고온 산화, 입계 편석",
}


# ==============================
# 추론 설정 (train_kaggle.py 가 저장한 config 와 동일한 기본값)
# ==============================

DEFAULT_INFER_CONFIG = {
    "img_size": 384,               # 패치를 모델에 넣을 때의 크기
    "patch_size": 384,             # 원본에서 자를 정사각 패치 한 변(px)
    "stride": 192,                 # 슬라이딩 윈도우 간격 (50% 겹침)
    "dilations": [1, 2, 4, 6],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# 기존(train.py, 224px) 가중치 — key 이름이 b1/b6/b12/b18
LEGACY_CONFIG = {
    "img_size": 224, "patch_size": None, "stride": None,
    "dilations": [1, 6, 12, 18],
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}


# ==============================
# ASPP 모듈 (train_kaggle.py 와 동일 구조)
# ==============================

class ASPP(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 256,
                 dilations=(1, 2, 4, 6)):
        super().__init__()

        def _branch(d: int):
            if d == 1:
                conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, out_channels, 3,
                                 padding=d, dilation=d, bias=False)
            return nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.GELU())

        self.branches = nn.ModuleList([_branch(d) for d in dilations])

        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        gap = F.interpolate(self.gap_branch(x), size=(h, w),
                            mode="bilinear", align_corners=False)
        return self.project(torch.cat([b(x) for b in self.branches] + [gap], dim=1))


# ==============================
# ConvNeXt-Small + ASPP 모델
# ==============================

class FractographyNet(nn.Module):
    """
    backbone → (B,768,12,12) @384px
    aspp     → (B,256,12,12)   ← GradCAM++ target
    """
    def __init__(self, num_classes: int = 4, dilations=(1, 2, 4, 6)):
        super().__init__()
        self.backbone = convnext_small(weights=None).features
        self.aspp = ASPP(in_channels=768, out_channels=256, dilations=dilations)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==============================
# pth 모델 로드 함수
# ==============================

_LEGACY_KEYS = {"aspp.b1.": "aspp.branches.0.", "aspp.b6.": "aspp.branches.1.",
                "aspp.b12.": "aspp.branches.2.", "aspp.b18.": "aspp.branches.3."}


def _read_config(model_path):
    """fractography_best.pth 옆의 *_config.json 을 찾아 읽음."""
    stem = os.path.splitext(model_path)[0]
    candidates = [stem + "_config.json",
                  os.path.join(os.path.dirname(model_path), "fractography_best_config.json")]
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return path, json.load(f)
    return None, None


def load_model(model_path: str, device, num_classes: int = 4):
    """
    반환: (model, infer_config)
      infer_config: img_size / patch_size / stride / dilations / mean / std
      - 새 가중치(train_kaggle.py): config.json 의 값 사용 (없으면 DEFAULT_INFER_CONFIG)
      - 기존 가중치(train.py, b1/b6/... key): LEGACY_CONFIG 로 자동 인식
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"pth 파일 없음: {model_path}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    legacy = any(k.startswith("aspp.b6.") for k in state)
    if legacy:
        cfg, cfg_path = dict(LEGACY_CONFIG), "(기존 train.py 가중치로 자동 인식)"
        for old, new in _LEGACY_KEYS.items():
            state = {k.replace(old, new): v for k, v in state.items()}
    else:
        cfg_path, file_cfg = _read_config(model_path)
        cfg = dict(DEFAULT_INFER_CONFIG)
        if file_cfg:
            cfg.update({k: file_cfg[k] for k in cfg if k in file_cfg})
        else:
            cfg_path = "(config.json 없음 → 기본값 사용)"

    model = FractographyNet(num_classes, dilations=tuple(cfg["dilations"])).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("✅ pth 로드 완료")
    print(f"✅ 경로: {model_path}")
    print(f"✅ 설정: {cfg_path}")
    print(f"✅ 입력 {cfg['img_size']}px | 패치 {cfg['patch_size']} / stride {cfg['stride']} "
          f"| dilations {cfg['dilations']}")
    print(f"✅ device: {device}")

    return model, cfg