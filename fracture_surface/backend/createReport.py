"""
보고서 생성 모듈 (/report)

LLM 분석(llm_service.py)의 결과를 받아 파손단면 AI 분석 보고서를 만든다.
공용 상수/유틸은 llm_service 에서 가져온다.
"""

import ollama
from datetime import datetime

from llm_service import (
    OLLAMA_MODEL,
    FALLBACK_CAUSE_EXPLANATION,
    KO_NAMES,
    MATERIAL_LABELS,
    build_cam_text,
    clean_text,
    ensure_sentence_end,
    get_confidence_level,
    parse_llm_json,
)


# =========================================================
# 보고서용 상수
# =========================================================

UNKNOWN_MATERIALS = ("", "unknown", None)

MATERIAL_TRAITS = {
    "steel": "강도와 인성의 균형이 좋지만 탄소량과 열처리에 따라 연성-취성 천이 온도가 나타날 수 있고, 반복 하중에서는 피로 한도가 존재하는 재료입니다.",
    "stainless_steel": "오스테나이트계는 인성이 높고 저온에서도 연성을 유지하지만 염화물 환경에서 응력부식균열에 민감할 수 있으며, 마르텐사이트계는 경도가 높은 대신 상대적으로 취성이 큽니다.",
    "aluminum": "가볍고 연성이 좋지만 뚜렷한 피로 한도가 없어 반복 하중에서 장기적으로 피로 균열이 발생할 수 있습니다.",
    "titanium": "비강도가 높지만 노치(응력 집중부)에 민감하고 수소를 흡수하면 취화될 수 있습니다.",
    "cast_iron": "내부 흑연 조직 때문에 인성이 낮고 인장에 약해 취성 파괴가 일어나기 쉬운 재료입니다.",
    "copper": "연성과 전도성이 높아 연성 파괴 경향이 크며, 조건에 따라 수소 취화가 문제될 수 있습니다.",
    "magnesium": "상온 성형성이 낮고 노치에 민감하며 부식과 피로에 상대적으로 취약한 편입니다.",
    "nickel_alloy": "고온 강도와 내식성이 우수하지만 고온 환경에서는 입계 석출물, 크리프와 관련된 입계 손상이 문제될 수 있습니다.",
    "tool_steel": "경도와 내마모성이 높은 대신 인성이 낮아 충격이나 응력 집중에 취성 파괴가 일어나기 쉽습니다.",
}

REPORT_TYPE_INFO = {
    "취성 파괴": "거의 소성 변형 없이 균열이 빠르게 진행되어 결정면을 따라 쪼개지는 파괴이며, 평탄하고 반짝이는 분리면이 특징입니다.",
    "연성 파괴": "큰 소성 변형 후 파단되는 파괴이며, 미세한 함몰(딤플) 패턴과 컵-콘 형태의 변형 흔적이 특징입니다.",
    "피로 파괴": "반복 하중으로 균열이 서서히 성장하다 최종 파단되는 파괴이며, 비치 마크(줄무늬)와 구역별 표면 차이가 특징입니다.",
    "입계 파괴": "결정립 경계를 따라 균열이 전파되는 파괴이며, 입자 경계가 드러난 표면이 특징입니다.",
}

NEXT_STEPS = {
    "취성 파괴": [
        {"title": "사용 온도와 충격 이력 확인", "detail": "파손 당시의 온도, 충격이나 급격한 하중 여부를 확인하세요. 저온이나 충격 조건은 취성 파괴의 주요 요인입니다."},
        {"title": "균열 시작점 추적", "detail": "파단면의 방사형(헤링본) 무늬를 따라 균열 시작점을 찾고, 시작점 주변에 노치, 용접부, 가공 흔적이 있는지 확인하세요."},
        {"title": "충격 시험", "detail": "샤르피 충격 시험으로 인성과 연성-취성 천이 온도를 확인하면 재료 자체의 취약 여부를 판단할 수 있습니다."},
        {"title": "미세조직 및 경도 확인", "detail": "단면 연마 후 조직 관찰과 경도 측정으로 열처리 상태, 개재물, 조직 불균일 여부를 확인하세요."},
    ],
    "연성 파괴": [
        {"title": "하중 이력 확인", "detail": "설계 하중을 초과한 과부하나 사고 기록이 있었는지 확인하세요."},
        {"title": "파단부 거시 관찰", "detail": "네킹(단면 수축), 컵-콘 형상, 전단립 방향을 확인하면 하중 방향과 변형 정도를 추정할 수 있습니다."},
        {"title": "SEM 딤플 관찰", "detail": "딤플의 크기와 형태(등축, 전단, 인장)를 관찰하면 하중 형태를 구분하는 데 도움이 됩니다."},
        {"title": "인장 시험과 단면 확인", "detail": "해당 재료의 강도와 연신율이 규격에 부합하는지, 실제 단면이 설계보다 줄어 있지 않은지 확인하세요."},
    ],
    "피로 파괴": [
        {"title": "균열 기점과 비치 마크 확인", "detail": "비치 마크가 시작되는 기점을 찾고, 기점이 표면인지 내부인지, 주변에 노치나 결함이 있는지 확인하세요."},
        {"title": "하중 이력 조사", "detail": "반복 횟수, 진동, 응력 진폭 등 운전 조건을 확인하세요. 최종 파단 영역의 비율로 하중 수준을 가늠할 수 있습니다."},
        {"title": "표면 상태 점검", "detail": "가공 흔적, 부식 피트, 용접부, 잔류 응력 등 기점 후보가 될 수 있는 표면 조건을 확인하세요."},
        {"title": "SEM 스트리에이션 관찰", "detail": "미세 줄무늬(스트리에이션)를 관찰하면 피로 파괴 여부를 더 확실하게 검증할 수 있습니다."},
    ],
    "입계 파괴": [
        {"title": "사용 환경 확인", "detail": "수소, 염화물이나 황화물 같은 부식 매질, 고온 노출 이력이 있었는지 확인하세요."},
        {"title": "SEM-EDS 성분 분석", "detail": "입계면에서 석출물이나 편석 원소(예: S, P 등)가 검출되는지 확인하세요."},
        {"title": "금속 조직 관찰", "detail": "단면을 연마하고 에칭해 결정립 크기, 입계 석출물, 조직 이상을 확인하세요."},
        {"title": "열처리·용접 이력 확인", "detail": "예민화, 템퍼 취화 등 입계를 약화시킬 수 있는 열처리나 용접 이력을 확인하세요."},
    ],
}

FORBIDDEN_DEFINITIVE = ["확실합니다", "분명합니다", "원인입니다", "틀림없습니다"]

MATERIAL_WORDS = [
    "재질", "스테인리스", "알루미늄", "티타늄", "주철",
    "구리", "마그네슘", "니켈", "공구강", "강재", "탄소강",
]


# =========================================================
# 보고서 보조 함수
# =========================================================

def build_next_steps(prediction, prediction_en, confidence_percent, material, gradcam_summary):

    steps = list(NEXT_STEPS[prediction])

    if material in UNKNOWN_MATERIALS:
        steps.append({
            "title": "재질 확인",
            "detail": "재질 정보가 없어 원인 해석에 반영하지 못했습니다. 성분 분석이나 경도 측정으로 재질을 확인하면 해석 정확도가 높아집니다.",
        })

    others = [
        KO_NAMES.get(k, k)
        for k in (gradcam_summary or {})
        if k != prediction_en
    ]

    if others:
        steps.append({
            "title": "함께 주목된 영역 확인",
            "detail": f"Grad-CAM++에서 {', '.join(others)} 관련 영역도 함께 주목되었습니다. 해당 부위를 확대 촬영해 별도로 관찰하세요.",
        })

    level = get_confidence_level(confidence_percent)

    if level == "low":
        steps.insert(0, {
            "title": "추가 이미지 촬영 및 전문가 검토",
            "detail": "예측 확률이 낮아 오분류 가능성이 있습니다. 다른 위치와 배율로 파단면을 추가 촬영하고 파손 해석 전문가의 검토를 받으세요.",
        })
    elif level == "medium":
        steps.append({
            "title": "추가 위치 촬영",
            "detail": "다른 위치와 배율의 이미지를 추가로 분석해 결과가 일관적인지 확인하세요.",
        })

    return steps


def build_confidence_comment(confidence_percent):

    level = get_confidence_level(confidence_percent)

    if level == "high":
        return (
            f"예측 확률은 {confidence_percent:.1f}%로 비교적 높은 편입니다. "
            "다만 이미지 분석만으로 실제 파손 원인을 확정할 수는 없으며, "
            "아래 확인 사항으로 검증하는 것이 좋습니다."
        )

    if level == "medium":
        return (
            f"예측 확률은 {confidence_percent:.1f}%로 중간 수준입니다. "
            "결과 해석에 주의가 필요하며 추가 이미지와 시험으로 보완하는 것이 좋습니다."
        )

    return (
        f"예측 확률은 {confidence_percent:.1f}%로 낮은 편입니다. "
        "오분류 가능성이 있으므로 추가 이미지와 전문가 검토가 필요합니다."
    )


def fallback_basis(prediction, prediction_en, gradcam_summary):

    info = (gradcam_summary or {}).get(prediction_en)

    if not info:
        return (
            f"Grad-CAM++에서 뚜렷한 활성화 영역이 확인되지 않아 "
            f"모델의 전체 출력 확률을 기준으로 {prediction}로 분류되었습니다."
        )

    return (
        f"Grad-CAM++ 분석에서 {prediction}에 해당하는 활성화 영역이 "
        f"{info['position']}에 {info['spread']} 형태로 나타났으며, "
        f"모델은 이 영역을 판단 근거로 활용한 것으로 해석할 수 있습니다."
    )


def check_report_text(text, fallback, *, other_types=None,
                      need_uncertain=False, no_material=False):

    text = clean_text(text)

    if not text:
        return fallback

    if any(w in text for w in FORBIDDEN_DEFINITIVE):
        print("[report validate] 단정 표현 → fallback")
        return fallback

    if other_types and any(o in text for o in other_types):
        print("[report validate] 다른 유형명 → fallback")
        return fallback

    if no_material and any(w in text for w in MATERIAL_WORDS):
        print("[report validate] 재질 미선택인데 재질 언급 → fallback")
        return fallback

    if need_uncertain and not any(
        w in text for w in ["가능성", "추정", "수 있습니다", "보입니다"]
    ):
        print("[report validate] 추정형 아님 → fallback")
        return fallback

    return ensure_sentence_end(text)


# =========================================================
# 보고서 Prompt
# =========================================================

def build_report_prompt(prediction, confidence_percent, material,
                        cam_text, feature, obs1, obs2):

    if material in UNKNOWN_MATERIALS:
        material_block = (
            "재질 정보 없음.\n"
            "재질이나 재질의 성질을 절대 언급하지 않는다. "
            "이미지 관찰과 Grad-CAM++ 결과만으로 서술한다."
        )
    else:
        material_ko = MATERIAL_LABELS.get(material, material)
        material_block = (
            f"사용자가 입력한 재질: {material_ko}\n"
            f"재질의 일반적 성질(참고): {MATERIAL_TRAITS.get(material, '')}\n"
            "cause 필드에서 이 재질 성질을 이미지 관찰, Grad-CAM++ 결과와 "
            "연결해 설명한다. 재질 성질만으로 원인을 단정하지 않는다."
        )

    return f"""
너는 금속 파손 단면 분석 결과를 보고서 문장으로 정리하는 도우미다.
파손 유형은 CNN이 이미 결정했으며 변경하거나 다른 유형을 제안하지 않는다.

[CNN 결과]
- 최종 파손 유형: {prediction}
- 신뢰도: {confidence_percent:.1f}%

[이전 이미지 분석 결과]
- 주요 특징: {feature}
- 관찰 1: {obs1}
- 관찰 2: {obs2}

[모델 판단 근거]
{cam_text}

[재질 정보]
{material_block}

[작성 규칙]
- 반드시 한국어로 작성한다.
- 입력에 없는 특징, 사용 환경, 사고 상황을 만들어내지 않는다.
- Grad-CAM++ 영역은 모델이 주목한 영역이며 실제 결함 위치가 아니다.
- 원인은 반드시 추정형("가능성이 있습니다", "추정할 수 있습니다")으로 쓴다.
- "확실합니다", "분명합니다", "원인입니다", "틀림없습니다"를 쓰지 않는다.
- 색상과 밝기를 특징으로 쓰지 않는다.

[필드]
- characteristics: "{prediction}"의 일반적인 특징과 이 이미지에서 관찰된 특징 (2~3문장)
- basis: Grad-CAM++ 영역(위치, 분포, 겹침)과 확률을 근거로 이렇게 분류된 이유 (2~4문장). 확률 수치를 나열하지 않는다.
- cause: 추정되는 파손 원인 (3~4문장). 앞선 특징과 근거를 연결한다.
- next_steps_note: 정확한 원인 분석을 위해 아래 확인 사항이 필요한 이유 (1문장)

JSON 이외의 문장은 출력하지 않는다.

{{
    "characteristics": "...",
    "basis": "...",
    "cause": "...",
    "next_steps_note": "..."
}}
"""


# =========================================================
# 보고서 생성
# =========================================================

def generate_report(prediction, prediction_en, confidence_percent,
                    material="", probabilities=None, gradcam_summary=None,
                    decision_path="", feature="", obs1="", obs2=""):

    no_material = material in UNKNOWN_MATERIALS
    other_types = [
        n for n in ["연성 파괴", "취성 파괴", "피로 파괴", "입계 파괴"]
        if n != prediction
    ]

    fb_characteristics = REPORT_TYPE_INFO[prediction]
    fb_basis = fallback_basis(prediction, prediction_en, gradcam_summary)
    fb_cause = FALLBACK_CAUSE_EXPLANATION[prediction]

    if not no_material and material in MATERIAL_TRAITS:
        fb_cause += " 재질 특성도 해석 시 참고할 수 있습니다."

    fb_note = "이미지만으로는 원인을 확정할 수 없어 아래 항목을 추가로 확인하는 것이 필요합니다."

    characteristics, basis, cause, note = (
        fb_characteristics, fb_basis, fb_cause, fb_note
    )

    cam_text = build_cam_text(gradcam_summary, probabilities, decision_path)

    prompt = build_report_prompt(
        prediction, confidence_percent, material,
        cam_text, feature, obs1, obs2,
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 금속 파손 분석 보고서 작성 도우미다. "
                        "CNN 결과를 변경하지 않고 입력된 정보만 사용한다. "
                        "원인은 추정형으로 쓰고 반드시 한국어 JSON만 출력한다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.25, "top_p": 0.9},
        )

        raw = response["message"]["content"].strip()
        print("Gemma 보고서 원본 응답:", raw)

        parsed = parse_llm_json(raw)

        if parsed:
            characteristics = check_report_text(
                parsed.get("characteristics", ""), fb_characteristics,
                other_types=other_types, no_material=no_material,
            )
            basis = check_report_text(
                parsed.get("basis", ""), fb_basis, no_material=no_material,
            )
            cause = check_report_text(
                parsed.get("cause", ""), fb_cause,
                other_types=other_types, need_uncertain=True,
                no_material=no_material,
            )
            note = check_report_text(
                parsed.get("next_steps_note", ""), fb_note,
                no_material=no_material,
            )
        else:
            print("보고서 JSON 파싱 실패 → fallback")

    except Exception as e:
        print(f"보고서 Gemma 오류: {e}")

    return {
        "characteristics": characteristics,
        "basis": basis,
        "cause": cause,
        "next_steps_note": note,
        "next_steps": build_next_steps(
            prediction, prediction_en, confidence_percent,
            material, gradcam_summary,
        ),
        "confidence_comment": build_confidence_comment(confidence_percent),
        "confidence_status": get_confidence_level(confidence_percent),
        "material_label": (
            "모름 (보고서에 반영하지 않음)" if no_material
            else MATERIAL_LABELS.get(material, material)
        ),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }