import { useState } from "react";

const CARD_INFO = {
  "취성 파괴": {
    en: "CLEAVAGE FRACTURE",
    color: "blue",
    cause: "충격 하중, 저온, 응력 집중",
    features: ["평평한 파단면", "빠른 균열 전파"],
    prevention: "인성 높은 재료 선택, 응력 집중 완화",
  },
  "연성 파괴": {
    en: "DUCTILE FRACTURE",
    color: "green",
    cause: "과도한 인장 하중",
    features: ["늘어난 흔적", "딤플 또는 컵-콘 형태"],
    prevention: "하중 관리, 적절한 단면 설계",
  },
  "피로 파괴": {
    en: "FATIGUE FRACTURE",
    color: "amber",
    cause: "장기간 반복 하중",
    features: ["줄무늬 형태", "균열의 점진적 성장"],
    prevention: "표면 처리, 응력 집중 최소화",
  },
  "입계 파괴": {
    en: "INTERGRANULAR FRACTURE",
    color: "red",
    cause: "고온, 부식, 재료 열화",
    features: ["결정립 경계 따라 균열 진행", "입자 경계가 드러난 표면"],
    prevention: "열처리 관리, 부식 환경 제어",
  },
};

const colorStyle = {
  blue: {
    border: "border-blue-300",
    text: "text-blue-700",
    badge: "bg-blue-100 text-blue-700",
    dot: "bg-blue-400",
    ring: "ring-blue-300",
    bg: "bg-blue-50",
  },
  green: {
    border: "border-green-300",
    text: "text-green-700",
    badge: "bg-green-100 text-green-700",
    dot: "bg-green-400",
    ring: "ring-green-300",
    bg: "bg-green-50",
  },
  amber: {
    border: "border-amber-300",
    text: "text-amber-700",
    badge: "bg-amber-100 text-amber-700",
    dot: "bg-amber-400",
    ring: "ring-amber-300",
    bg: "bg-amber-50",
  },
  red: {
    border: "border-red-300",
    text: "text-red-700",
    badge: "bg-red-100 text-red-700",
    dot: "bg-red-400",
    ring: "ring-red-300",
    bg: "bg-red-50",
  },
};

export default function FlipCard({
  type,
  similarity,
  isBest,
  isMixed,
  mixedMode,
  imageSlot,
}) {
  const [flipped, setFlipped] = useState(false);

  const info = CARD_INFO[type];
  const style = colorStyle[info.color];

  const highlightClass = isMixed
    ? `ring-4 ${style.ring} ${style.bg} scale-[1.02] shadow-lg`
    : "bg-white shadow-sm hover:shadow-md";

  const renderEnglishTitle = () => (
    <p className={`text-xs font-bold tracking-[0.18em] leading-4 ${style.text}`}>
      {info.en.split(" ").map((word) => (
        <span key={word} className="block">
          {word}
        </span>
      ))}
    </p>
  );

  return (
    <div
      onClick={() => setFlipped(!flipped)}
      className={`rounded-2xl border ${style.border} p-5 h-[420px] cursor-pointer transition ${highlightClass}`}
    >
      {!flipped ? (
        <div className="h-full flex flex-col">
          <div className="h-[78px] flex items-start justify-between gap-3">
            <div className="min-w-0">
              {renderEnglishTitle()}
              <h3 className="text-xl font-bold mt-1 leading-7">{type}</h3>
            </div>

            <span
              className={`shrink-0 min-w-[86px] text-center px-3 py-1 rounded-full text-sm font-bold ${style.badge}`}
            >
              유사도 {similarity}
            </span>
          </div>

          <div className="h-[180px] rounded-xl overflow-hidden bg-slate-100">
            {imageSlot}
          </div>

          <div className="flex-1 pt-4 space-y-2">
            {isBest && !mixedMode && (
              <p className={`text-sm font-semibold ${style.text}`}>
                입력 이미지와 가장 유사한 유형입니다.
              </p>
            )}

            {isMixed && mixedMode && (
              <p className={`text-sm font-semibold ${style.text}`}>
                혼합 가능성이 있는 주요 후보 유형입니다.
              </p>
            )}
          </div>

          <p className="text-xs text-slate-400 text-right">
            클릭하면 상세 설명
          </p>
        </div>
      ) : (
        <div className="h-full flex flex-col">
          <div className="h-[78px] flex items-start justify-between gap-3">
            <div className="min-w-0">
              {renderEnglishTitle()}
              <h3 className="text-xl font-bold mt-1 leading-7">{type}</h3>
            </div>

            <span
              className={`shrink-0 min-w-[86px] text-center px-3 py-1 rounded-full text-sm font-bold ${style.badge}`}
            >
              유사도 {similarity}
            </span>
          </div>

          <div className="flex-1 space-y-4 text-sm text-slate-700 leading-6 overflow-y-auto pr-2">
            <div>
              <p className="font-semibold text-slate-500 mb-1">주요 원인</p>
              <p>{info.cause}</p>
            </div>

            <div>
              <p className="font-semibold text-slate-500 mb-1">표면 특징</p>
              <ul className="space-y-1">
                {info.features.map((feature) => (
                  <li key={feature} className="flex gap-2">
                    <span
                      className={`mt-2 w-1.5 h-1.5 rounded-full shrink-0 ${style.dot}`}
                    />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="pt-3 border-t border-slate-200">
              <p className="font-semibold text-slate-500 mb-1">예방 대책</p>
              <p>{info.prevention}</p>
            </div>
          </div>

          <p className="pt-4 text-xs text-slate-400 text-right">
            클릭하면 되돌아가기
          </p>
        </div>
      )}
    </div>
  );
}