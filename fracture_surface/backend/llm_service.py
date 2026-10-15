import re
import json
import ollama


# =========================================================
# Ollama 모델 설정
# =========================================================

OLLAMA_MODEL = "gemma3:4b"


# =========================================================
# fallback용 기본 정보
# =========================================================

FALLBACK_FEATURES = {
    "취성 파괴": "표면의 균열 및 제한적인 변형 형태",
    "연성 파괴": "표면의 함몰 및 소성 변형 형태",
    "피로 파괴": "영역별 표면 차이 및 균열 성장 형태",
    "입계 파괴": "입자형 또는 경계형 분리 구조",
}


EXPECTED_CAUSE = {
    "취성 파괴": "급격한 응력 집중 가능성",
    "연성 파괴": "국부적인 하중 집중 가능성",
    "피로 파괴": "장기간 반복 하중 가능성",
    "입계 파괴": "결정립 경계 약화 가능성",
}


# =========================================================
# 유형별 허용 원인 후보
# =========================================================

CAUSE_CANDIDATES = {
    "취성 파괴": [
        "급격한 응력 집중 가능성",
        "충격성 하중의 영향 가능성",
        "국부적인 응력 집중에 따른 빠른 균열 진행 가능성",
    ],

    "연성 파괴": [
        "국부적인 하중 집중 가능성",
        "과도한 하중에 따른 변형 가능성",
        "국부적인 변형 집중 가능성",
        "과도한 변형이 파손에 영향을 주었을 가능성",
    ],

    "피로 파괴": [
        "장기간 반복 하중 가능성",
        "반복 응력에 따른 균열 성장 가능성",
        "국부적인 반복 응력이 균열 성장에 관여했을 가능성",
        "반복적인 하중에 따른 점진적 균열 전파 가능성",
    ],

    "입계 파괴": [
        "결정립 경계 약화 가능성",
        "결정립 경계 부근의 취약화 가능성",
        "입계 영역의 약화가 파손에 영향을 주었을 가능성",
    ],
}


# =========================================================
# 메커니즘 fallback
# =========================================================

FALLBACK_MECHANISM = {
    "취성 파괴":
        "이러한 표면 형태는 CNN이 예측한 취성 파괴에서 "
        "제한적인 소성 변형과 함께 균열이 빠르게 진행되는 "
        "메커니즘과 관련해 해석할 수 있습니다.",

    "연성 파괴":
        "이러한 표면 형태는 CNN이 예측한 연성 파괴에서 "
        "소성 변형이 동반되는 파손 메커니즘과 "
        "관련해 해석할 수 있습니다.",

    "피로 파괴":
        "이러한 표면의 영역별 차이는 CNN이 예측한 피로 파괴에서 "
        "균열이 점진적으로 성장하는 메커니즘과 "
        "관련해 해석할 수 있습니다.",

    "입계 파괴":
        "이러한 표면 구조는 CNN이 예측한 입계 파괴에서 "
        "결정립 경계를 따라 분리가 진행되는 메커니즘과 "
        "관련해 해석할 수 있습니다.",
}


# =========================================================
# 원인 설명 fallback
# =========================================================

FALLBACK_CAUSE_EXPLANATION = {
    "취성 파괴":
        "가능한 원인으로는 급격한 응력 집중이나 충격성 하중이 "
        "파손에 영향을 주었을 가능성이 있습니다.",

    "연성 파괴":
        "가능한 원인으로는 국부적인 하중 집중이나 과도한 변형이 "
        "파손에 영향을 주었을 가능성이 있습니다.",

    "피로 파괴":
        "가능한 원인으로는 장기간의 반복 하중이나 반복 응력이 "
        "균열 성장에 영향을 주었을 가능성이 있습니다.",

    "입계 파괴":
        "가능한 원인으로는 결정립 경계의 약화가 "
        "파손에 영향을 주었을 가능성이 있습니다.",
}


# =========================================================
# 전체 설명 fallback
# =========================================================

RULE_BASED_EXPLANATIONS = {
    "취성 파괴":
        "파단면에서 표면의 불규칙한 형상과 균열과 관련된 구조가 관찰될 수 있습니다. "
        "구체적인 표면 형태는 영역에 따라 서로 다르게 나타날 수 있습니다. "
        + FALLBACK_MECHANISM["취성 파괴"] + " "
        + FALLBACK_CAUSE_EXPLANATION["취성 파괴"],

    "연성 파괴":
        "파단면에서 함몰되거나 변형된 형태가 관찰될 수 있습니다. "
        "이러한 구조는 영역에 따라 크기와 분포가 서로 다르게 나타날 수 있습니다. "
        + FALLBACK_MECHANISM["연성 파괴"] + " "
        + FALLBACK_CAUSE_EXPLANATION["연성 파괴"],

    "피로 파괴":
        "파단면에서 서로 다른 표면 영역과 균열 성장과 관련된 형태가 관찰될 수 있습니다. "
        "구조의 분포나 방향성은 이미지 영역에 따라 달라질 수 있습니다. "
        + FALLBACK_MECHANISM["피로 파괴"] + " "
        + FALLBACK_CAUSE_EXPLANATION["피로 파괴"],

    "입계 파괴":
        "파단면에서 입자형 또는 경계형 구조가 관찰될 수 있습니다. "
        "이러한 구조의 크기와 분포는 영역에 따라 다르게 나타날 수 있습니다. "
        + FALLBACK_MECHANISM["입계 파괴"] + " "
        + FALLBACK_CAUSE_EXPLANATION["입계 파괴"],
}


# =========================================================
# 재질 정보
# =========================================================

MATERIAL_LABELS = {
    "steel": "강",
    "stainless_steel": "스테인리스강",
    "aluminum": "알루미늄",
    "titanium": "티타늄",
    "cast_iron": "주철",
    "copper": "구리",
    "magnesium": "마그네슘 합금",
    "nickel_alloy": "니켈 합금",
    "tool_steel": "공구강",
    "unknown": "정보 없음",
    "": "미선택",
}


# =========================================================
# 텍스트 처리
# =========================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = str(text)

    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("#", "")

    text = re.sub(
        r"^\s*\d+\.\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"^\s*[-•]\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_llm_json(text: str):

    try:
        return json.loads(text)

    except Exception:
        pass

    try:
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if match:
            return json.loads(
                match.group()
            )

    except Exception:
        pass

    return None


def ensure_sentence_end(text: str) -> str:

    text = clean_text(text)

    if not text:
        return ""

    if text[-1] not in ".!?。":
        text += "."

    return text


# =========================================================
# 신뢰도
# =========================================================

def get_confidence_level(
    confidence_percent: float
) -> str:

    if confidence_percent < 60:
        return "low"

    elif confidence_percent < 80:
        return "medium"

    return "high"


def get_confidence_instruction(
    confidence_percent: float
) -> str:

    level = get_confidence_level(
        confidence_percent
    )

    if level == "low":
        return (
            "CNN 신뢰도가 낮다. "
            "파손 유형과 이미지 특징의 관계를 강하게 단정하지 않는다. "
            "'관련될 가능성이 있습니다', "
            "'해석할 수 있습니다'와 같은 표현을 사용한다. "
            "최종 설명에 추가 검토 필요성을 포함한다."
        )

    if level == "medium":
        return (
            "CNN 신뢰도가 중간 수준이다. "
            "확정적인 표현을 피하고 "
            "'관련될 가능성이 있습니다', "
            "'해석할 수 있습니다'와 같이 작성한다."
        )

    return (
        "CNN 신뢰도가 비교적 높더라도 "
        "이미지만으로 파손 메커니즘이나 원인을 확정하지 않는다. "
        "'관련성이 높아 보입니다', "
        "'해석할 수 있습니다', "
        "'가능성이 있습니다'와 같이 작성한다."
    )


# =========================================================
# 재질 입력
# =========================================================

def build_condition_text(
    material: str
) -> str:

    if material in ["", "unknown", None]:
        return "재질 정보 없음"

    material_ko = MATERIAL_LABELS.get(
        material,
        material,
    )

    return (
        f"사용자가 입력한 참고 재질: {material_ko}\n"
        "재질은 참고 메타데이터일 뿐이며 "
        "이미지 관찰, 주요 특징, 파손 메커니즘, "
        "파손 원인을 결정하는 근거로 사용하지 않는다."
    )


# =========================================================
# 단일 분석 Prompt
# =========================================================

def build_prompt(
    prediction: str,
    confidence_percent: float,
    material: str,
) -> str:

    condition_text = build_condition_text(
        material
    )

    confidence_instruction = (
        get_confidence_instruction(
            confidence_percent
        )
    )

    cause_candidates_text = "\n".join(
        f"- {cause}"
        for cause in CAUSE_CANDIDATES[
            prediction
        ]
    )

    return f"""
너는 금속 파손 단면 이미지의 시각적 특징을 관찰하고,
CNN 분석 결과를 일반 사용자가 이해하기 쉽게 설명하는 도우미다.

실제 파손 단면 이미지가 함께 제공된다.


==================================================
[역할 분담]
==================================================

파손 유형은 별도의 CNN 모델이 이미 결정하였다.

CNN 결과:
- 최종 예측 파손 유형: {prediction}
- CNN 신뢰도: {confidence_percent:.1f}%

너는 파손 유형을 다시 분류하지 않는다.
CNN의 최종 예측 결과 "{prediction}"을 유지한다.

너의 역할:

1. 현재 이미지를 독립적으로 관찰한다.
2. 이미지에서 가장 중요한 시각적 특징을 추출한다.
3. 관찰 특징과 CNN 예측 파손 메커니즘의 관계를 설명한다.
4. 가능한 파손 원인을 허용된 후보 범위 안에서 선택한다.

다른 파손 유형을 제안하거나
CNN 결과를 변경해서는 안 된다.


==================================================
[재질 정보]
==================================================

{condition_text}

재질 사용 규칙:

- 재질을 보고 파손 유형을 추정하지 않는다.
- 재질을 보고 주요 특징을 작성하지 않는다.
- 재질을 보고 파손 원인을 선택하지 않는다.
- 재질의 일반적인 기계적 성질을 임의로 추가하지 않는다.
- 설명에 필요하지 않으면 재질명을 언급하지 않는다.


==================================================
[이미지 우선 관찰]
==================================================

CNN 예측 유형에 대한 지식을 사용하기 전에
먼저 현재 이미지 자체를 관찰한다.

현재 이미지에서 실제로 확인되는 것만 설명한다.

관찰 가능한 요소:

- 표면의 평탄함 또는 거칠기
- 함몰되거나 돌출된 구조
- 구조의 크기
- 구조의 밀도
- 구조의 분포
- 구조의 방향성
- 중앙과 가장자리의 차이
- 특정 위치에 집중된 특징
- 표면 영역별 차이
- 선형 또는 방사형 구조
- 찢어지거나 늘어진 형태
- 입자 또는 경계처럼 나뉜 구조
- 반복되거나 불규칙한 표면 패턴

모든 항목을 사용할 필요는 없다.
현재 이미지에서 실제로 눈에 띄는 특징만 선택한다.


==================================================
[색상 및 촬영 조건]
==================================================

색상, 밝기, 명암 차이는
촬영 환경이나 조명에 의해 달라질 수 있으므로
주요 파손 특징으로 사용하지 않는다.

형상, 위치, 크기, 밀도, 분포,
방향성, 영역별 구조 차이를 우선한다.


==================================================
[feature]
==================================================

현재 이미지에서 가장 눈에 띄는 특징을
15~40자 정도의 짧은 구절로 작성한다.

같은 파손 유형이라도 이미지가 다르면
feature가 달라질 수 있어야 한다.

가능하면 다음을 조합한다.

- 위치
- 대표 구조
- 크기
- 밀도
- 분포
- 방향성

예시 형식:

"중앙부에 조밀하게 분포한 미세 함몰"
"우측 영역에 집중된 불규칙 선형 구조"
"표면 전반에 분산된 크기 다양한 요철"

현재 이미지에 없는 특징을 사용하지 않는다.

feature에는 다음을 넣지 않는다.

- 파손 유형 이름
- CNN 결과
- 파손 원인
- 하중
- 응력
- 재질 이름
- 색상
- 밝기


==================================================
[visual_observation_1]
==================================================

현재 이미지에서 가장 눈에 띄는
시각적 특징을 한 문장으로 작성한다.

형상 중심으로 작성하며,
파손 유형의 일반적 특징을
실제 관찰처럼 만들어내지 않는다.


==================================================
[visual_observation_2]
==================================================

첫 번째 관찰을 더 구체적으로 설명한다.

실제로 보이는 경우에만 다음을 사용한다.

- 위치
- 크기
- 밀도
- 분포
- 방향성
- 균일성
- 중앙과 주변의 차이
- 영역별 차이

visual_observation_1을 반복하지 않는다.


==================================================
[mechanism_relation]
==================================================

visual_observation_1과 visual_observation_2에서
관찰한 특징과 CNN이 예측한 "{prediction}"의
파손 메커니즘 관계만 설명한다.

여기에서는 파손 원인을 설명하지 않는다.

다음 표현을 사용하지 않는다.

- 하중
- 반복 하중
- 과도한 하중
- 응력
- 반복 응력
- 응력 집중
- 원인
- 사용 환경

유형별 허용 메커니즘:

연성 파괴:
- 소성 변형
- 미세 공극의 형성 및 결합
- 변형을 동반한 파단 과정

취성 파괴:
- 제한적인 소성 변형
- 빠른 균열 진행
- 균열을 따라 분리되는 과정

피로 파괴:
- 균열의 점진적인 성장
- 균열 전파
- 영역별 균열 성장 차이

입계 파괴:
- 결정립 경계를 따른 분리
- 입자 경계 형태의 분리 과정

다른 파손 유형을 언급하지 않는다.


==================================================
[expected_cause]
==================================================

현재 이미지에서 관찰한 표면 특징과
CNN이 예측한 파손 유형을 함께 참고한다.

이미지만으로 실제 사고 원인을 확정할 수 없으므로
아래 후보 중 가장 자연스럽게 연결되는
원인 표현 하나만 선택한다.

항상 첫 번째 후보를 선택하지 않는다.

현재 "{prediction}"의 허용 원인 후보:

{cause_candidates_text}


==================================================
[원인 후보 선택 기준]
==================================================

원인 후보는 단순히 파손 유형만 보고 선택하지 않는다.

이미지에서 관찰된

- 구조가 집중된 위치
- 구조의 분포 범위
- 크기 차이
- 밀도 차이
- 특정 영역과 주변 영역의 차이
- 변형 구조의 국부성 또는 전체 분포

를 참고한다.


연성 파괴:

- 특정 영역에 함몰·변형 구조가 집중됨
  → 국부적인 하중 집중 가능성
  → 국부적인 변형 집중 가능성

- 변형 구조가 표면 전체에 넓게 분포
  → 과도한 하중에 따른 변형 가능성

- 특정 위치의 크기나 밀도가 주변과 다름
  → 국부적인 변형 집중 가능성


피로 파괴:

- 넓은 영역에서 점진적으로 이어짐
  → 장기간 반복 하중 가능성
  → 반복적인 하중에 따른 점진적 균열 전파 가능성

- 특정 영역에 균열 전파가 집중됨
  → 국부적인 반복 응력이 균열 성장에 관여했을 가능성

- 영역별 표면 차이와 점진적 성장 형태가 함께 나타남
  → 반복 응력에 따른 균열 성장 가능성


취성 파괴:

- 국부적인 균열·분리 형태가 집중됨
  → 급격한 응력 집중 가능성

- 급격한 분리 형태가 넓게 나타남
  → 충격성 하중의 영향 가능성

- 특정 영역에서 균열 진행 형태가 두드러짐
  → 국부적인 응력 집중에 따른 빠른 균열 진행 가능성


입계 파괴:

- 입자 경계 형태가 넓게 나타남
  → 결정립 경계 약화 가능성

- 특정 영역에 경계형 구조가 집중됨
  → 결정립 경계 부근의 취약화 가능성

- 입계 형태의 분리가 특정 영역에서 두드러짐
  → 입계 영역의 약화가 파손에 영향을 주었을 가능성


==================================================
[중요]
==================================================

위 기준은 참고 규칙이다.

이미지에서 해당 특징이 실제로 확인되는 경우에만 적용한다.

다음 정보는 입력되지 않았으므로 추측하지 않는다.

- 사용 환경
- 부품 종류
- 온도
- 열처리
- 부식
- 실제 하중 크기
- 사고 상황
- 진동
- 회전체
- 반복 굽힘
- 충돌 사고


==================================================
[cause_explanation]
==================================================

expected_cause에서 선택한 원인을
한 문장으로 설명한다.

원인 설명은 이 필드에서만 수행한다.

반드시 추정형으로 작성한다.

예:

"가능성이 있습니다."
"관련될 수 있습니다."
"영향을 주었을 가능성이 있습니다."
"추정할 수 있습니다."


==================================================
[신뢰도]
==================================================

{confidence_instruction}


==================================================
[이미지별 차별화]
==================================================

같은 파손 유형이라도
각 이미지를 새롭게 관찰한다.

다음 중 실제로 두드러지는 항목을 기준으로
이미지별 특징을 차별화한다.

- 구조의 크기
- 특정 영역 집중 여부
- 분포
- 방향성
- 밀도
- 영역별 차이

문장을 다양하게 만들기 위해
이미지에 없는 특징을 만들지 않는다.


==================================================
[문체]
==================================================

- 반드시 한국어로 작성한다.
- 일반 사용자가 이해하기 쉽게 작성한다.
- 이미지에 없는 내용을 만들지 않는다.
- 같은 의미를 반복하지 않는다.
- 확정적인 표현을 사용하지 않는다.

금지 표현:

"확실합니다"
"분명합니다"
"원인입니다"
"틀림없습니다"


==================================================
[출력]
==================================================

JSON 이외의 문장은 절대 출력하지 않는다.

반드시 아래 JSON 형식으로 출력한다.

{{
    "feature": "현재 이미지에서 가장 눈에 띄는 주요 특징",
    "visual_observation_1": "이미지 대표 특징 한 문장",
    "visual_observation_2": "위치·크기·밀도·분포 등을 구체화한 한 문장",
    "mechanism_relation": "관찰 특징과 CNN 예측 파손 메커니즘의 관계 한 문장",
    "expected_cause": "허용 후보 중 선택한 짧은 원인",
    "cause_explanation": "가능한 파손 원인을 설명하는 한 문장"
}}
"""


# =========================================================
# feature 검증
# =========================================================

def validate_feature(
    prediction: str,
    feature: str,
) -> str:

    feature = clean_text(feature)

    if not feature:
        return FALLBACK_FEATURES[prediction]

    feature = feature.rstrip(".!?。")

    fracture_names = [
        "연성 파괴",
        "취성 파괴",
        "피로 파괴",
        "입계 파괴",
    ]

    for name in fracture_names:
        feature = feature.replace(
            name,
            "",
        )

    feature = feature.replace(
        "CNN",
        "",
    )

    feature = feature.replace(
        "예측",
        "",
    )

    color_keywords = [
        "푸른색",
        "파란색",
        "붉은색",
        "빨간색",
        "색상",
        "밝기",
        "어두운",
        "밝은",
    ]

    if any(
        word in feature
        for word in color_keywords
    ):
        print(
            "[feature validate] "
            "색상 중심 특징 감지 → fallback"
        )

        return FALLBACK_FEATURES[
            prediction
        ]

    cause_keywords = [
        "하중",
        "응력",
        "원인",
        "피로 누적",
    ]

    if any(
        word in feature
        for word in cause_keywords
    ):
        print(
            "[feature validate] "
            "원인 표현 감지 → fallback"
        )

        return FALLBACK_FEATURES[
            prediction
        ]

    feature = re.sub(
        r"\s+",
        " ",
        feature,
    ).strip()

    if len(feature) < 4:
        return FALLBACK_FEATURES[
            prediction
        ]

    return feature


# =========================================================
# 관찰 문장 검증
# =========================================================

def validate_visual_observation(
    text: str,
) -> str:

    text = ensure_sentence_end(
        text
    )

    if not text:
        return ""

    return text


# =========================================================
# 메커니즘 문장 검증
# =========================================================

def validate_mechanism_relation(
    prediction: str,
    text: str,
) -> str:

    text = clean_text(
        text
    )

    if not text:
        return FALLBACK_MECHANISM[
            prediction
        ]

    other_predictions = [
        name
        for name in [
            "연성 파괴",
            "취성 파괴",
            "피로 파괴",
            "입계 파괴",
        ]
        if name != prediction
    ]

    for other in other_predictions:

        if other in text:

            print(
                "[mechanism validate] "
                f"다른 파손 유형 감지: {other}"
            )

            return FALLBACK_MECHANISM[
                prediction
            ]

    cause_keywords = [
        "하중",
        "응력",
        "원인",
        "사용 환경",
    ]

    if any(
        word in text
        for word in cause_keywords
    ):
        print(
            "[mechanism validate] "
            "원인 관련 표현 감지 → fallback"
        )

        return FALLBACK_MECHANISM[
            prediction
        ]

    mechanism_keywords = {
        "연성 파괴": [
            "소성 변형",
            "공극",
            "변형",
            "파단",
        ],

        "취성 파괴": [
            "균열",
            "소성 변형",
            "분리",
        ],

        "피로 파괴": [
            "균열",
            "성장",
            "전파",
            "점진",
        ],

        "입계 파괴": [
            "결정립",
            "경계",
            "분리",
        ],
    }

    if not any(
        word in text
        for word in mechanism_keywords[
            prediction
        ]
    ):
        print(
            "[mechanism validate] "
            "메커니즘 표현 부족 → fallback"
        )

        return FALLBACK_MECHANISM[
            prediction
        ]

    return ensure_sentence_end(
        text
    )


# =========================================================
# expected_cause 검증
# =========================================================

def validate_expected_cause(
    prediction: str,
    cause: str,
) -> str:

    cause = clean_text(
        cause
    )

    if not cause:
        return EXPECTED_CAUSE[
            prediction
        ]

    fracture_names = [
        "연성 파괴",
        "취성 파괴",
        "피로 파괴",
        "입계 파괴",
    ]

    for name in fracture_names:
        cause = cause.replace(
            name,
            "",
        )

    cause = re.sub(
        r"\s+",
        " ",
        cause,
    ).strip()

    invalid_keywords = {
        "연성 파괴": [
            "반복 하중",
            "반복적인 하중",
            "반복 응력",
            "반복적인 응력",
            "피로",
            "결정립 경계",
        ],

        "취성 파괴": [
            "반복 하중",
            "반복적인 하중",
            "반복 응력",
            "반복적인 응력",
            "피로",
            "큰 소성 변형",
        ],

        "피로 파괴": [
            "단일 하중",
            "한 번의 큰 하중",
            "결정립 경계 약화",
            "과도한 소성 변형",
        ],

        "입계 파괴": [
            "반복 하중",
            "반복적인 하중",
            "반복 응력",
            "반복적인 응력",
            "피로",
            "큰 소성 변형",
        ],
    }

    for word in invalid_keywords.get(
        prediction,
        [],
    ):

        if word in cause:

            print(
                "[cause validate] "
                f"부적절 원인 감지: {word}"
            )

            return EXPECTED_CAUSE[
                prediction
            ]

    candidate_keywords = {
        "취성 파괴": [
            "응력 집중",
            "충격",
        ],

        "연성 파괴": [
            "하중 집중",
            "과도한 하중",
            "변형 집중",
            "과도한 변형",
        ],

        "피로 파괴": [
            "반복 하중",
            "반복 응력",
            "균열 성장",
            "균열 전파",
        ],

        "입계 파괴": [
            "결정립 경계",
            "입계",
            "취약화",
            "약화",
        ],
    }

    if not any(
        word in cause
        for word in candidate_keywords[
            prediction
        ]
    ):
        print(
            "[cause validate] "
            "허용 원인 범위와 불일치 → fallback"
        )

        return EXPECTED_CAUSE[
            prediction
        ]

    return cause


# =========================================================
# cause_explanation 검증
# =========================================================

def validate_cause_explanation(
    prediction: str,
    text: str,
) -> str:

    text = clean_text(
        text
    )

    if not text:
        return FALLBACK_CAUSE_EXPLANATION[
            prediction
        ]

    fracture_names = [
        "연성 파괴",
        "취성 파괴",
        "피로 파괴",
        "입계 파괴",
    ]

    for name in fracture_names:

        if name in text:

            print(
                "[cause explanation validate] "
                "파손 유형명 포함 → fallback"
            )

            return FALLBACK_CAUSE_EXPLANATION[
                prediction
            ]

    invalid_keywords = {
        "연성 파괴": [
            "반복 하중",
            "반복적인 하중",
            "반복 응력",
            "반복적인 응력",
            "피로",
            "결정립 경계",
        ],

        "취성 파괴": [
            "반복 하중",
            "반복적인 하중",
            "반복 응력",
            "반복적인 응력",
            "피로",
            "큰 소성 변형",
        ],

        "피로 파괴": [
            "한 번의 큰 하중",
            "급격한 단일 하중",
            "결정립 경계 약화",
        ],

        "입계 파괴": [
            "반복 하중",
            "반복적인 하중",
            "반복 응력",
            "반복적인 응력",
            "피로",
            "큰 소성 변형",
        ],
    }

    for word in invalid_keywords.get(
        prediction,
        [],
    ):

        if word in text:

            print(
                "[cause explanation validate] "
                f"부적절 원인 감지: {word}"
            )

            return FALLBACK_CAUSE_EXPLANATION[
                prediction
            ]

    uncertainty_words = [
        "가능성이 있습니다",
        "관련될 수 있습니다",
        "영향을 주었을 가능성이 있습니다",
        "추정할 수 있습니다",
        "가능성이",
        "추정",
    ]

    if not any(
        word in text
        for word in uncertainty_words
    ):
        print(
            "[cause explanation validate] "
            "추정형 표현 없음 → fallback"
        )

        return FALLBACK_CAUSE_EXPLANATION[
            prediction
        ]

    return ensure_sentence_end(
        text
    )


# =========================================================
# 관찰 문장 두 개 확보
# =========================================================

def build_visual_sentences(
    prediction: str,
    feature: str,
    observation_1: str,
    observation_2: str,
):

    observation_1 = (
        validate_visual_observation(
            observation_1
        )
    )

    observation_2 = (
        validate_visual_observation(
            observation_2
        )
    )

    if not observation_1:

        observation_1 = (
            f"현재 이미지에서는 {feature} 형태가 "
            "주요한 표면 특징으로 관찰됩니다."
        )

    if not observation_2:

        observation_2 = (
            "해당 구조는 이미지 내 위치에 따라 "
            "크기와 분포에서 차이를 보입니다."
        )

    if observation_1 == observation_2:

        observation_2 = (
            "해당 구조는 이미지 내 위치에 따라 "
            "크기와 분포에서 차이를 보입니다."
        )

    return (
        ensure_sentence_end(
            observation_1
        ),
        ensure_sentence_end(
            observation_2
        ),
    )


# =========================================================
# 최종 explanation 조립
# =========================================================

def compose_explanation(
    prediction: str,
    confidence_percent: float,
    visual_observation_1: str,
    visual_observation_2: str,
    mechanism_relation: str,
    cause_explanation: str,
) -> str:

    sentences = [
        ensure_sentence_end(
            visual_observation_1
        ),

        ensure_sentence_end(
            visual_observation_2
        ),

        ensure_sentence_end(
            mechanism_relation
        ),

        ensure_sentence_end(
            cause_explanation
        ),
    ]

    if get_confidence_level(
        confidence_percent
    ) == "low":

        sentences.append(
            "CNN 신뢰도가 낮아 추가 이미지나 "
            "전문가 검토를 함께 고려하는 것이 좋습니다."
        )

    sentences = [
        sentence
        for sentence in sentences
        if sentence
    ]

    return " ".join(
        sentences
    ).strip()


# =========================================================
# fallback
# =========================================================

def fallback_analysis(
    prediction: str,
    confidence_percent: float,
):

    base = RULE_BASED_EXPLANATIONS[
        prediction
    ]

    if get_confidence_level(
        confidence_percent
    ) == "low":

        return (
            base
            + " CNN 신뢰도가 낮아 추가 이미지나 "
              "전문가 검토를 함께 고려하는 것이 좋습니다."
        )

    return base


# =========================================================
# 단일 이미지 Gemma 멀티모달 분석
# =========================================================

def generate_llm_analysis(
    prediction: str,
    confidence_percent: float,
    material: str = "",
    image_bytes: bytes = None,
):

    # =====================================================
    # 먼저 모든 값을 fallback으로 초기화
    # → JSON 실패 / Ollama 오류가 나도 안전하게 반환 가능
    # =====================================================

    image_feature = (
        FALLBACK_FEATURES[
            prediction
        ]
    )

    observation_1 = (
        f"현재 이미지에서는 {image_feature}가 "
        "주요한 표면 특징으로 관찰됩니다."
    )

    observation_2 = (
        "해당 구조는 이미지 내 위치에 따라 "
        "크기와 분포에서 차이를 보입니다."
    )

    mechanism_relation = (
        FALLBACK_MECHANISM[
            prediction
        ]
    )

    llm_expected_cause = (
        EXPECTED_CAUSE[
            prediction
        ]
    )

    cause_explanation = (
        FALLBACK_CAUSE_EXPLANATION[
            prediction
        ]
    )

    korean_explanation = (
        fallback_analysis(
            prediction,
            confidence_percent,
        )
    )

    prompt = build_prompt(
        prediction,
        confidence_percent,
        material,
    )

    try:

        has_image = (
            image_bytes is not None
        )

        if has_image:
            print(
                "[Gemma] 이미지 바이트 입력 사용"
            )

        else:
            print(
                "[Gemma] 이미지 없음 → "
                "텍스트 정보만 사용"
            )

        user_message = {
            "role": "user",
            "content": prompt,
        }

        if has_image:

            user_message[
                "images"
            ] = [
                image_bytes
            ]

        response = ollama.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "system",

                    "content": (
                        "너는 금속 파손 단면 이미지의 시각적 특징을 "
                        "관찰하는 멀티모달 설명 생성기다. "
                        "파손 유형은 CNN이 이미 결정했으며 변경하지 않는다. "
                        "먼저 현재 이미지를 독립적으로 관찰한다. "
                        "위치, 크기, 밀도, 분포, 방향성, 영역별 차이를 우선한다. "
                        "색상이나 밝기를 주요 특징으로 사용하지 않는다. "
                        "메커니즘과 원인을 분리해서 설명한다. "
                        "원인은 허용 후보 범위 안에서만 선택한다. "
                        "이미지에 없는 특징이나 사용 환경을 만들어내지 않는다. "
                        "재질은 판단 근거로 사용하지 않는다. "
                        "반드시 한국어 JSON만 출력한다."
                    ),
                },

                user_message,
            ],

            options={
                "temperature": 0.25,
                "top_p": 0.9,
            },
        )

        raw_response = (
            response[
                "message"
            ][
                "content"
            ].strip()
        )

        print(
            "Gemma 원본 응답:",
            raw_response,
        )

        parsed = parse_llm_json(
            raw_response
        )

        # =================================================
        # JSON 정상
        # =================================================

        if parsed:

            image_feature = (
                validate_feature(
                    prediction,
                    parsed.get(
                        "feature",
                        "",
                    ),
                )
            )

            observation_1 = clean_text(
                parsed.get(
                    "visual_observation_1",
                    "",
                )
            )

            observation_2 = clean_text(
                parsed.get(
                    "visual_observation_2",
                    "",
                )
            )

            (
                observation_1,
                observation_2,
            ) = build_visual_sentences(
                prediction,
                image_feature,
                observation_1,
                observation_2,
            )

            mechanism_relation = (
                validate_mechanism_relation(
                    prediction,
                    parsed.get(
                        "mechanism_relation",
                        "",
                    ),
                )
            )

            llm_expected_cause = (
                validate_expected_cause(
                    prediction,
                    parsed.get(
                        "expected_cause",
                        "",
                    ),
                )
            )

            cause_explanation = (
                validate_cause_explanation(
                    prediction,
                    parsed.get(
                        "cause_explanation",
                        "",
                    ),
                )
            )

            korean_explanation = (
                compose_explanation(
                    prediction,
                    confidence_percent,
                    observation_1,
                    observation_2,
                    mechanism_relation,
                    cause_explanation,
                )
            )

            print(
                "[Gemma] 최종 주요 특징:",
                image_feature,
            )

            print(
                "[Gemma] 관찰 1:",
                observation_1,
            )

            print(
                "[Gemma] 관찰 2:",
                observation_2,
            )

            print(
                "[Gemma] 메커니즘:",
                mechanism_relation,
            )

            print(
                "[Gemma] 예상 원인:",
                llm_expected_cause,
            )

            print(
                "[Gemma] 원인 설명:",
                cause_explanation,
            )

        else:

            print(
                "JSON 파싱 실패 → fallback 사용"
            )

    except Exception as e:

        print(
            f"Gemma / Ollama 오류: {e}"
        )

    # =====================================================
    # 비교 기능에서도 사용하도록 세부 필드 모두 반환
    # =====================================================

    return {
        "feature":
            image_feature,

        "visual_observation_1":
            observation_1,

        "visual_observation_2":
            observation_2,

        "mechanism_relation":
            mechanism_relation,

        "cause":
            llm_expected_cause,

        "expected_cause":
            llm_expected_cause,

        "cause_explanation":
            cause_explanation,

        "explanation":
            korean_explanation,
    }


# =========================================================
# 분석 결과 비교
# =========================================================

def generate_compare_analysis(
    compare_items: list
):

    def parse_confidence(value):

        if isinstance(
            value,
            str,
        ):

            value = (
                value
                .replace(
                    "%",
                    "",
                )
                .strip()
            )

        try:
            return float(
                value
            )

        except Exception:
            return 0.0


    if len(compare_items) < 2:

        return {
            "summary":
                "비교를 위해서는 두 개 이상의 분석 결과가 필요합니다.",

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
        }


    item1 = compare_items[0]
    item2 = compare_items[1]


    # =====================================================
    # 기본 분석 정보
    # =====================================================

    pred1 = item1.get(
        "prediction",
        "-"
    )

    pred2 = item2.get(
        "prediction",
        "-"
    )

    conf1 = parse_confidence(
        item1.get(
            "confidence",
            0,
        )
    )

    conf2 = parse_confidence(
        item2.get(
            "confidence",
            0,
        )
    )


    feature1 = item1.get(
        "feature",
        "-"
    )

    feature2 = item2.get(
        "feature",
        "-"
    )


    obs1_1 = item1.get(
        "visual_observation_1",
        ""
    )

    obs1_2 = item1.get(
        "visual_observation_2",
        ""
    )

    obs2_1 = item2.get(
        "visual_observation_1",
        ""
    )

    obs2_2 = item2.get(
        "visual_observation_2",
        ""
    )


    mechanism1 = item1.get(
        "mechanism_relation",
        ""
    )

    mechanism2 = item2.get(
        "mechanism_relation",
        ""
    )


    cause1 = item1.get(
        "expected_cause",
        "-"
    )

    cause2 = item2.get(
        "expected_cause",
        "-"
    )


    cause_exp1 = item1.get(
        "cause_explanation",
        ""
    )

    cause_exp2 = item2.get(
        "cause_explanation",
        ""
    )


    # =====================================================
    # 기존 분석 기록과 호환
    # =====================================================

    if not obs1_1:
        obs1_1 = item1.get(
            "explanation",
            "-"
        )

    if not obs2_1:
        obs2_1 = item2.get(
            "explanation",
            "-"
        )


    # =====================================================
    # 신뢰도는 Python에서 직접 계산
    # =====================================================

    conf_gap = abs(
        conf1 - conf2
    )


    if conf_gap < 5:

        confidence_text = (
            f"분석 1은 {conf1:.1f}%, "
            f"분석 2는 {conf2:.1f}%로 "
            f"차이는 {conf_gap:.1f}%p이며 "
            "두 결과의 신뢰도는 유사한 수준입니다."
        )

    elif conf1 > conf2:

        confidence_text = (
            f"분석 1은 {conf1:.1f}%, "
            f"분석 2는 {conf2:.1f}%로 "
            f"분석 1이 {conf_gap:.1f}%p 높습니다."
        )

    else:

        confidence_text = (
            f"분석 1은 {conf1:.1f}%, "
            f"분석 2는 {conf2:.1f}%로 "
            f"분석 2가 {conf_gap:.1f}%p 높습니다."
        )


    if conf1 < 60 or conf2 < 60:

        confidence_text += (
            " 60% 미만의 결과가 포함되어 있어 "
            "해석 시 추가 검토가 필요할 수 있습니다."
        )


    # =====================================================
    # 동일 유형 / 다른 유형에 따른 비교 기준
    # =====================================================

    if pred1 == pred2:

        type_instruction = f"""
두 분석 모두 CNN이 "{pred1}"으로 예측하였다.

따라서 파손 유형 자체의 차이를 만들어내지 않는다.

같은 파손 유형 안에서

- 표면 형상
- 구조의 위치
- 크기
- 밀도
- 분포
- 방향성
- 영역별 차이
- 메커니즘 표현
- 예상 원인

이 어떻게 다른지를 중심으로 비교한다.
"""

    else:

        type_instruction = f"""
CNN 예측 결과가 서로 다르다.

분석 1: {pred1}
분석 2: {pred2}

두 파손 유형을 변경하거나 재분류하지 않는다.

각 이미지에서 실제 관찰된 특징을 기준으로
두 파손 유형에서 나타난 구조와
메커니즘 차이를 설명한다.
"""


    # =====================================================
    # 비교 Prompt
    # =====================================================

    prompt = f"""
너는 금속 파손 단면 분석 결과 두 개를
서로 비교하여 설명하는 도우미다.

중요:
너는 이미지를 다시 분류하지 않는다.
CNN이 결정한 파손 유형을 그대로 유지한다.

두 분석 결과에 실제로 존재하는 정보만 사용한다.


==================================================
[분석 1]
==================================================

파손 유형:
{pred1}

CNN 신뢰도:
{conf1:.1f}%

주요 특징:
{feature1}

관찰 특징 1:
{obs1_1}

관찰 특징 2:
{obs1_2}

메커니즘 관계:
{mechanism1}

예상 원인:
{cause1}

원인 설명:
{cause_exp1}


==================================================
[분석 2]
==================================================

파손 유형:
{pred2}

CNN 신뢰도:
{conf2:.1f}%

주요 특징:
{feature2}

관찰 특징 1:
{obs2_1}

관찰 특징 2:
{obs2_2}

메커니즘 관계:
{mechanism2}

예상 원인:
{cause2}

원인 설명:
{cause_exp2}


==================================================
[파손 유형 비교 조건]
==================================================

{type_instruction}


==================================================
[비교 우선순위]
==================================================

다음 순서로 비교한다.

1. 실제 관찰된 표면 형상
2. 구조의 위치
3. 구조의 크기
4. 구조의 밀도
5. 구조의 분포
6. 구조의 방향성
7. 영역별 차이
8. CNN 파손 메커니즘과의 관계
9. 예상 원인

입력 결과에 없는 특징은 추가하지 않는다.

차이가 명확하지 않은 항목은
억지로 차이를 만들지 않는다.


==================================================
[공통점]
==================================================

두 분석 결과에 실제로 공통적으로 나타난 특징만 설명한다.

파손 유형이 같다는 이유만으로
일반적 특징을 만들어내지 않는다.


==================================================
[시각적 차이]
==================================================

두 이미지의 관찰 결과를 직접 비교한다.

단순히 두 결과를 각각 나열하지 않는다.

가능하면

"분석 1은 ~인 반면,
분석 2는 ~"

형태로 직접 비교한다.


==================================================
[메커니즘 차이]
==================================================

입력된 mechanism_relation을 중심으로
각 이미지의 관찰 특징과
CNN 파손 메커니즘 관계의 차이를 설명한다.

여기에서는 파손 원인을 설명하지 않는다.


==================================================
[예상 원인 차이]
==================================================

각 분석에서 이미 제시된 expected_cause만 비교한다.

새로운 사고 원인을 생성하지 않는다.

두 원인이 동일하면 동일하다고 명확하게 설명한다.

예상 원인은 실제 사고 원인이 아니라
가능성을 나타내는 참고 정보로 설명한다.


==================================================
[신뢰도]
==================================================

신뢰도 계산 결과:

{confidence_text}

이 값을 변경하거나 다시 계산하지 않는다.

신뢰도를 이미지 품질이나
실제 파손 원인의 확실성과 동일시하지 않는다.


==================================================
[최종 해석]
==================================================

같은 유형이면
같은 파손 유형 안에서 어떤 관찰 차이가 있는지 설명한다.

다른 유형이면
각 파손 유형에서 어떤 관찰 차이가 나타나는지 설명한다.

이미지만으로 실제 사고 원인을 확정하지 않는다.


==================================================
[금지 사항]
==================================================

- CNN 결과 변경
- 새로운 파손 유형 제안
- 입력에 없는 이미지 특징 생성
- 실제 사고 상황 추측
- 부품 종류 추측
- 사용 환경 추측
- 재질 특성으로 차이 생성
- 원인 단정
- 동일 내용 반복


==================================================
[출력]
==================================================

반드시 한국어 JSON만 출력한다.

{{
    "summary": "두 분석의 가장 중요한 차이를 한 문장으로 요약",
    "common_point": "두 결과의 실제 공통점",
    "visual_difference": "두 이미지에서 관찰된 표면 특징의 직접적인 차이",
    "mechanism_difference": "관찰 특징과 파손 메커니즘 관계의 차이",
    "cause_difference": "두 분석에서 제시된 예상 원인의 차이",
    "final_opinion": "전체 비교에 대한 1~2문장 해석"
}}
"""


    # =====================================================
    # fallback
    # =====================================================

    fallback_result = {
        "summary":
            "두 분석 결과는 이미지에서 관찰된 표면 형태와 분포를 중심으로 비교할 수 있습니다.",

        "common_point":
            "두 결과 모두 이미지에서 관찰된 표면 특징을 바탕으로 CNN 결과를 해석하고 있습니다.",

        "visual_difference":
            "두 이미지의 세부적인 표면 구조와 분포를 함께 비교할 필요가 있습니다.",

        "mechanism_difference":
            "각 이미지에서 관찰된 특징과 CNN 파손 메커니즘의 관계를 비교할 수 있습니다.",

        "cause_difference":
            "예상 원인은 이미지 분석을 바탕으로 제시된 가능성이므로 참고 정보로 해석하는 것이 좋습니다.",

        "confidence_difference":
            confidence_text,

        "final_opinion":
            "두 결과의 관찰 특징과 CNN 분석을 함께 비교하여 해석하는 것이 좋으며, "
            "이미지만으로 실제 파손 원인을 확정하기는 어렵습니다.",
    }


    try:

        response = ollama.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "system",

                    "content": (
                        "너는 금속 파손 단면 분석 결과 두 개를 "
                        "비교하는 설명 생성기다. "
                        "CNN 결과를 변경하지 않는다. "
                        "각 이미지에서 이미 추출된 관찰 특징을 직접 비교한다. "
                        "두 결과를 따로 나열하지 말고 "
                        "공통점과 차이를 명확하게 제시한다. "
                        "입력에 없는 이미지 특징이나 사고 상황을 생성하지 않는다. "
                        "메커니즘과 원인을 구분하여 설명한다. "
                        "반드시 한국어 JSON만 출력한다."
                    ),
                },

                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            options={
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )

        raw_response = (
            response[
                "message"
            ][
                "content"
            ].strip()
        )

        print(
            "Gemma 비교 원본 응답:",
            raw_response,
        )

        parsed = parse_llm_json(
            raw_response
        )

        if not parsed:
            return fallback_result


        return {
            "summary":
                clean_text(
                    parsed.get(
                        "summary",
                        fallback_result[
                            "summary"
                        ],
                    )
                ),

            "common_point":
                clean_text(
                    parsed.get(
                        "common_point",
                        fallback_result[
                            "common_point"
                        ],
                    )
                ),

            "visual_difference":
                clean_text(
                    parsed.get(
                        "visual_difference",
                        fallback_result[
                            "visual_difference"
                        ],
                    )
                ),

            "mechanism_difference":
                clean_text(
                    parsed.get(
                        "mechanism_difference",
                        fallback_result[
                            "mechanism_difference"
                        ],
                    )
                ),

            "cause_difference":
                clean_text(
                    parsed.get(
                        "cause_difference",
                        fallback_result[
                            "cause_difference"
                        ],
                    )
                ),

            # 신뢰도는 LLM이 생성하지 않고
            # Python에서 계산한 문장을 그대로 사용
            "confidence_difference":
                confidence_text,

            "final_opinion":
                clean_text(
                    parsed.get(
                        "final_opinion",
                        fallback_result[
                            "final_opinion"
                        ],
                    )
                ),
        }


    except Exception as e:

        print(
            f"Gemma 비교 분석 오류: {e}"
        )

        return fallback_result
