import { useState } from "react";

/**
 * RAG 참고 문헌 카드
 *
 * 백엔드 /analyze 응답의 references 를 그대로 받아 표시한다.
 *   references: [{ source, page, score, snippet }, ...]
 *
 * - references 가 없으면(undefined / 이전 분석 기록) 아무것도 그리지 않는다.
 * - references 가 빈 배열이면 "검색된 문헌 없음" 안내를 보여준다.
 *
 * 문헌을 추가했다면 아래 SOURCE_INFO 에 파일명 → 표시 제목/링크를 적어 두면
 * 카드에 파일명 대신 논문 제목이 표시된다. (없으면 파일명 그대로 표시)
 */

const SOURCE_INFO = {
  "metals-09-00148.pdf": {
    title: "Pantazopoulos (2019), Metals 9(2), 148",
    url: "https://www.mdpi.com/2075-4701/9/2/148",
  },
};

function ReferenceItem({ item, index }) {
  const [open, setOpen] = useState(false);

  const info = SOURCE_INFO[item.source];
  const title = info?.title || item.source || "출처 미상";

  const score =
    typeof item.score === "number" ? item.score.toFixed(2) : null;

  const snippet = (item.snippet || "").trim();

  return (
    <li className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0 text-sm">
          <span className="mr-2 text-xs font-bold text-emerald-700">
            [{index + 1}]
          </span>

          {info?.url ? (
            <a
              href={info.url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-semibold text-slate-800 underline decoration-slate-300 underline-offset-2 hover:text-emerald-700"
            >
              {title}
            </a>
          ) : (
            <span className="font-semibold text-slate-800">{title}</span>
          )}

          {item.page ? (
            <span className="ml-2 text-xs text-slate-500">p.{item.page}</span>
          ) : null}
        </div>

        {score ? (
          <span className="shrink-0 text-xs text-slate-500">
            검색 유사도 {score}
          </span>
        ) : null}
      </div>

      {snippet ? (
        <>
          <p
            className={`mt-2 text-sm leading-6 text-slate-600 ${
              open ? "" : "line-clamp-3"
            }`}
            lang="en"
          >
            “{snippet}
            {snippet.length >= 200 ? "…" : ""}”
          </p>

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="mt-1 text-xs font-semibold text-emerald-700 hover:underline"
          >
            {open ? "접기" : "펼쳐 보기"}
          </button>
        </>
      ) : null}
    </li>
  );
}

export default function ReferenceCard({ references, number = 4 }) {
  // 서버가 references 를 보내지 않은 이전 분석 기록은 카드를 숨긴다
  if (!Array.isArray(references)) return null;

  return (
    <div className="rounded-xl border border-slate-200 p-5">
      <div className="mb-3 flex items-center gap-3">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-600 text-xs font-bold text-white">
          {number}
        </span>

        <h4 className="font-bold">관련 문헌 검색 결과</h4>
      </div>

      {references.length > 0 ? (
        <>
          <p className="mb-3 text-xs leading-5 text-slate-500">
            예측된 파손 유형과 관련해 문헌에서 검색된 발췌입니다(영어 원문).
            AI 설명의 배경 지식으로 함께 제공되었으며, 이미지 판독 결과를
            대체하지 않습니다.
          </p>

          <ul className="space-y-3">
            {references.map((item, index) => (
              <ReferenceItem
                key={`${item.source}-${item.page}-${index}`}
                item={item}
                index={index}
              />
            ))}
          </ul>
        </>
      ) : (
        <p className="text-sm leading-6 text-slate-500">
          이번 분석에서는 검색된 참고 문헌이 없습니다. 문헌 인덱스가 아직
          만들어지지 않았거나, 관련도가 기준에 못 미친 경우입니다.
        </p>
      )}
    </div>
  );
}
