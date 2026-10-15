import {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";

import { addHistoryItem, fetchHistory, clearAllHistory } from "./firebase";

import {
  buildPhaseHistoryItem,
  addPhaseHistoryItem,
  fetchPhaseHistory,
  renamePhaseHistoryItem,
  clearAllPhaseHistory,
} from "./firebasePhase";

const CLASS_COLORS = {
  Cleavage: "#2563EB",
  Ductile: "#16A34A",
  Fatigue: "#D97706",
  Intergranular: "#DC2626",
};

const EN_NAMES = {
  "취성 파괴": "Cleavage",
  "연성 파괴": "Ductile",
  "피로 파괴": "Fatigue",
  "입계 파괴": "Intergranular",
};

const MATERIAL_LABELS = {
  steel: "강 (Steel)",
  stainless_steel: "스테인리스강",
  aluminum: "알루미늄",
  titanium: "티타늄",
  cast_iron: "주철",
  copper: "구리",
  magnesium: "마그네슘 합금",
  nickel_alloy: "니켈 합금",
  tool_steel: "공구강",
  unknown: "모름",
};

const CONFIDENCE_BOX_STYLES = {
  high: "border-green-200 bg-green-50 text-green-700",
  medium: "border-amber-200 bg-amber-50 text-amber-700",
  low: "border-red-200 bg-red-50 text-red-700",
};

const layout = {
  page: "min-h-screen bg-[#f4f7fb] text-slate-900",
  container: "max-w-[1500px] mx-auto px-6",
};

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function decodeMaskFromImage(img, W, H) {
  const c = document.createElement("canvas");

  c.width = W;
  c.height = H;

  const ctx = c.getContext("2d");

  ctx.drawImage(img, 0, 0, W, H);

  const { data } = ctx.getImageData(0, 0, W, H);

  const mask = new Uint8Array(W * H);

  for (let i = 0; i < W * H; i++) {
    mask[i] = data[i * 4] > 127 ? 255 : 0;
  }

  return mask;
}

function maskToSegments(mask, W, H) {
  const segments = [];

  const get = (x, y) =>
    x < 0 || y < 0 || x >= W || y >= H
      ? 0
      : mask[y * W + x] > 0
      ? 1
      : 0;

  for (let y = -1; y < H; y++) {
    for (let x = -1; x < W; x++) {
      const tl = get(x, y);
      const tr = get(x + 1, y);
      const bl = get(x, y + 1);
      const br = get(x + 1, y + 1);

      const code =
        (tl << 3) |
        (tr << 2) |
        (br << 1) |
        bl;

      const top = {
        x: x + 1.0,
        y: y + 0.5,
      };

      const right = {
        x: x + 1.5,
        y: y + 1.0,
      };

      const bottom = {
        x: x + 1.0,
        y: y + 1.5,
      };

      const left = {
        x: x + 0.5,
        y: y + 1.0,
      };

      switch (code) {
        case 1:
          segments.push([left, bottom]);
          break;

        case 2:
          segments.push([bottom, right]);
          break;

        case 3:
          segments.push([left, right]);
          break;

        case 4:
          segments.push([top, right]);
          break;

        case 5:
          segments.push([left, top]);
          segments.push([bottom, right]);
          break;

        case 6:
          segments.push([top, bottom]);
          break;

        case 7:
          segments.push([left, top]);
          break;

        case 8:
          segments.push([top, left]);
          break;

        case 9:
          segments.push([top, bottom]);
          break;

        case 10:
          segments.push([top, right]);
          segments.push([left, bottom]);
          break;

        case 11:
          segments.push([top, right]);
          break;

        case 12:
          segments.push([left, right]);
          break;

        case 13:
          segments.push([bottom, right]);
          break;

        case 14:
          segments.push([left, bottom]);
          break;

        default:
          break;
      }
    }
  }

  return segments;
}

function segmentsToPolylines(segments) {
  const key = (p) =>
    `${p.x.toFixed(2)},${p.y.toFixed(2)}`;

  const map = new Map();

  segments.forEach((seg, i) => {
    const k1 = key(seg[0]);
    const k2 = key(seg[1]);

    if (!map.has(k1)) {
      map.set(k1, []);
    }

    if (!map.has(k2)) {
      map.set(k2, []);
    }

    map.get(k1).push(i);
    map.get(k2).push(i);
  });

  const used = new Array(
    segments.length
  ).fill(false);

  const polylines = [];

  for (
    let i = 0;
    i < segments.length;
    i++
  ) {
    if (used[i]) {
      continue;
    }

    used[i] = true;

    const poly = [
      segments[i][0],
      segments[i][1],
    ];

    let extended = true;

    while (extended) {
      extended = false;

      const tail =
        poly[poly.length - 1];

      const cands =
        map.get(key(tail)) || [];

      for (const ci of cands) {
        if (used[ci]) {
          continue;
        }

        const seg =
          segments[ci];

        if (
          key(seg[0]) ===
          key(tail)
        ) {
          poly.push(seg[1]);
          used[ci] = true;
          extended = true;
          break;
        }

        if (
          key(seg[1]) ===
          key(tail)
        ) {
          poly.push(seg[0]);
          used[ci] = true;
          extended = true;
          break;
        }
      }
    }

    polylines.push(poly);
  }

  return polylines;
}

function drawPolylines(
  ctx,
  polylines,
  color,
  {
    dashed = false,
    lineWidth = 3,
    dashPattern = null,
    dashOffset = 0,
  } = {}
) {
  ctx.save();

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";

  const isDashed =
    !!dashPattern || dashed;

  ctx.lineCap = isDashed
    ? "butt"
    : "round";

  if (dashPattern) {
    ctx.setLineDash(
      dashPattern
    );

    ctx.lineDashOffset =
      dashOffset;
  } else if (dashed) {
    ctx.setLineDash([
      16,
      10,
    ]);

    ctx.lineDashOffset = 0;
  } else {
    ctx.setLineDash([]);

    ctx.lineDashOffset = 0;
  }

  for (const poly of polylines) {
    if (poly.length < 2) {
      continue;
    }

    ctx.beginPath();

    ctx.moveTo(
      poly[0].x,
      poly[0].y
    );

    for (
      let i = 1;
      i < poly.length;
      i++
    ) {
      ctx.lineTo(
        poly[i].x,
        poly[i].y
      );
    }

    ctx.stroke();
  }

  ctx.restore();
}

// 좌표 JSON(gradcam_contours)을 캔버스에 그릴 수 있는 폴리라인으로 변환
function contoursToPolylines(contoursForClass) {
  return contoursForClass.map((c) => {
    const poly = c.pts.map(([x, y]) => ({ x, y }));
    if (poly.length > 1) {
      poly.push(poly[0]); // 닫힌 도형으로 만들기
    }
    return { poly, overlaps: c.overlaps_with || [] };
  });
}

function GradcamView({
  result,
  chipSize = "text-xs",
  canvasClass = "h-[320px]",
}) {
  const canvasRef =
    useRef(null);

  const baseImgRef =
    useRef(null);

  const masksRef =
    useRef({});

  const layerImgsRef =
    useRef({});

    const hasMasks =
    !!result.gradcam_masks;

  const hasLayers =
    !hasMasks && !!result.gradcam_layers;

  const hasContoursOnly =
    !hasMasks &&
    !hasLayers &&
    !!result.gradcam_contours &&
    Object.keys(result.gradcam_contours).length > 0;

  const sourceObj =
    useMemo(() => {
      if (hasMasks) return result.gradcam_masks;
      if (hasLayers) return result.gradcam_layers;
      return result.gradcam_contours || {};
    }, [
      hasMasks,
      hasLayers,
      result.gradcam_masks,
      result.gradcam_layers,
      result.gradcam_contours,
    ]);

  const allClasses =
    useMemo(() => {
      return Object.keys(
        sourceObj
      );
    }, [sourceObj]);

  const activeClasses =
    useMemo(() => {
      return allClasses.filter(
        (name) => {
          const contours =
            result
              .gradcam_contours?.[
              name
            ];

          return (
            contours &&
            contours.length > 0
          );
        }
      );
    }, [
      allClasses,
      result.gradcam_contours,
    ]);

  const [checked, setChecked] =
    useState(() =>
      Object.fromEntries(
        allClasses.map(
          (name) => [
            name,
            true,
          ]
        )
      )
    );

  const redraw =
    useCallback(() => {
      const canvas =
        canvasRef.current;

      const base =
        baseImgRef.current;

      if (
        !canvas ||
        !base
      ) {
        return;
      }

      const W =
        base.naturalWidth;

      const H =
        base.naturalHeight;

      canvas.width = W;
      canvas.height = H;

      const ctx =
        canvas.getContext("2d");

      ctx.clearRect(
        0,
        0,
        W,
        H
      );

      ctx.drawImage(
        base,
        0,
        0
      );

      if (hasMasks) {
        const active =
          allClasses.filter(
            (name) =>
              checked[name]
          );

        if (
          active.length === 0
        ) {
          return;
        }

        const masks = {};

        for (
          const name of active
        ) {
          if (
            masksRef.current[
              name
            ]
          ) {
            masks[name] =
              masksRef.current[
                name
              ];
          }
        }

        const DASH_ON = 14;
        const SLOT = 22;

        const isInsideMask = (
          x,
          y,
          name
        ) => {
          const m =
            masks[name];

          if (!m) {
            return false;
          }

          const ix =
            Math.round(x);

          const iy =
            Math.round(y);

          if (
            ix < 0 ||
            iy < 0 ||
            ix >= W ||
            iy >= H
          ) {
            return false;
          }

          return (
            m[
              iy * W +
                ix
            ] > 0
          );
        };

        const othersContaining =
          (
            p,
            selfName
          ) => {
            const found =
              [];

            for (
              const other of
              active
            ) {
              if (
                other ===
                selfName
              ) {
                continue;
              }

              if (
                isInsideMask(
                  p.x,
                  p.y,
                  other
                )
              ) {
                found.push(
                  other
                );
              }
            }

            return found;
          };

        for (
          const name of active
        ) {
          const mask =
            masks[name];

          if (!mask) {
            continue;
          }

          const layerCanvas =
            document.createElement(
              "canvas"
            );

          layerCanvas.width =
            W;

          layerCanvas.height =
            H;

          const layerCtx =
            layerCanvas.getContext(
              "2d"
            );

          const segs =
            maskToSegments(
              mask,
              W,
              H
            );

          if (
            segs.length ===
            0
          ) {
            ctx.drawImage(
              layerCanvas,
              0,
              0
            );

            continue;
          }

          const polylines =
            segmentsToPolylines(
              segs
            );

          for (
            const poly of
            polylines
          ) {
            if (
              poly.length < 2
            ) {
              continue;
            }

            const pointStates =
              poly.map(
                (p) => {
                  const others =
                    othersContaining(
                      p,
                      name
                    );

                  if (
                    others.length ===
                    0
                  ) {
                    return "solo";
                  }

                  return (
                    "overlap:" +
                    others
                      .sort()
                      .join(",")
                  );
                }
              );

            let segStart = 0;

            for (
              let i = 1;
              i <=
              pointStates.length;
              i++
            ) {
              const isEnd =
                i ===
                pointStates.length;

              const stateChanged =
                !isEnd &&
                pointStates[
                  i
                ] !==
                  pointStates[
                    segStart
                  ];

              if (
                isEnd ||
                stateChanged
              ) {
                const subPoly =
                  poly.slice(
                    segStart,
                    i +
                      (isEnd
                        ? 0
                        : 1)
                  );

                const state =
                  pointStates[
                    segStart
                  ];

                if (
                  state ===
                  "solo"
                ) {
                  drawPolylines(
                    layerCtx,
                    [
                      subPoly,
                    ],
                    CLASS_COLORS[
                      name
                    ],
                    {
                      lineWidth: 3,
                    }
                  );
                } else {
                  const others =
                    state
                      .slice(
                        "overlap:"
                          .length
                      )
                      .split(",");

                  const candidates =
                    [
                      name,
                      ...others,
                    ].sort();

                  const N =
                    candidates.length;

                  const myIndex =
                    candidates.indexOf(
                      name
                    );

                  const period =
                    N *
                    SLOT;

                  const dashPattern =
                    [
                      DASH_ON,
                      period -
                        DASH_ON,
                    ];

                  const dashOffset =
                    -myIndex *
                    SLOT;

                  drawPolylines(
                    layerCtx,
                    [
                      subPoly,
                    ],
                    CLASS_COLORS[
                      name
                    ],
                    {
                      lineWidth: 4.5,
                      dashPattern,
                      dashOffset,
                    }
                  );
                }

                segStart = i;
              }
            }
          }

          ctx.drawImage(
            layerCanvas,
            0,
            0
          );
        }
      } else if (hasContoursOnly) {
        for (
          const name of
          allClasses
        ) {
          if (!checked[name]) continue;

          const contoursForClass =
            result.gradcam_contours?.[name] || [];

          if (contoursForClass.length === 0) continue;

          const items = contoursToPolylines(contoursForClass);

          const solo = items
            .filter((it) => it.overlaps.length === 0)
            .map((it) => it.poly);

          const overlap = items
            .filter((it) => it.overlaps.length > 0)
            .map((it) => it.poly);

          if (solo.length > 0) {
            drawPolylines(ctx, solo, CLASS_COLORS[name], {
              lineWidth: 3,
            });
          }

          if (overlap.length > 0) {
            drawPolylines(ctx, overlap, CLASS_COLORS[name], {
              dashed: true,
              lineWidth: 4.5,
            });
          }
        }
      } else {
        for (
          const name of
          allClasses
        ) {
          if (!checked[name]) continue;
          const layerImg = layerImgsRef.current[name];
          if (layerImg) {
            ctx.drawImage(layerImg, 0, 0);
          }
        }
      }
    }, [
      allClasses,
      checked,
      hasMasks,
      hasContoursOnly,
      result.gradcam_contours,
    ]);

  useEffect(() => {
    setChecked(
      Object.fromEntries(
        allClasses.map(
          (name) => [
            name,
            true,
          ]
        )
      )
    );
  }, [allClasses]);

  useEffect(() => {
    let cancelled =
      false;

    const loadAll =
      async () => {
        if (
          !result.base_image
        ) {
          return;
        }

        const base =
          await loadImage(
            result.base_image
          );

        if (cancelled) {
          return;
        }

        baseImgRef.current =
          base;

        const W =
          base.naturalWidth;

        const H =
          base.naturalHeight;

        if (hasMasks) {
          masksRef.current = {};
          for (const name of allClasses) {
            const src = sourceObj[name];
            if (!src) continue;
            const img = await loadImage(src);
            if (cancelled) return;
            masksRef.current[name] = decodeMaskFromImage(img, W, H);
          }
        } else if (hasContoursOnly) {
          // 좌표 데이터는 이미 result 안에 있어 별도 이미지 로딩이 필요 없음
        } else {
          layerImgsRef.current = {};
          for (const name of allClasses) {
            const src = sourceObj[name];
            if (!src) continue;
            layerImgsRef.current[name] = await loadImage(src);
            if (cancelled) return;
          }
        }

        redraw();
      };

    loadAll();

    return () => {
      cancelled = true;
    };
  }, [
    result.base_image,
    hasMasks,
    hasContoursOnly,  
    allClasses,
    sourceObj,
    redraw,
  ]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  const toggle = (name) => {
    setChecked(
      (prev) => ({
        ...prev,
        [name]:
          !prev[name],
      })
    );
  };

  return (
    <div>
      {activeClasses.length >
        0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {activeClasses.map(
            (name) => {
              const koName =
                Object.keys(
                  EN_NAMES
                ).find(
                  (key) =>
                    EN_NAMES[
                      key
                    ] ===
                    name
                );

              const color =
                CLASS_COLORS[
                  name
                ];

              const on =
                checked[
                  name
                ];

              return (
                <button
                  key={
                    name
                  }
                  onClick={() =>
                    toggle(
                      name
                    )
                  }
                  style={{
                    borderColor:
                      color,

                    backgroundColor:
                      on
                        ? color
                        : "transparent",

                    color:
                      on
                        ? "#fff"
                        : color,
                  }}
                  className={`px-3 py-1 rounded-full font-semibold border-2 transition ${chipSize}`}
                >
                  {koName ||
                    name}
                </button>
              );
            }
          )}
        </div>
      )}

      <div
        className={`w-full ${canvasClass} rounded-xl border border-slate-200 bg-white overflow-hidden flex items-center justify-center`}
      >
        <canvas
          ref={
            canvasRef
          }
          className="w-full h-full object-contain"
          style={{
            display:
              "block",
          }}
        />
      </div>
    </div>
  );
}

function ModalShell({
  title,
  subtitle,
  onClose,
  children,
  maxWidth = "max-w-5xl",
}) {
  return (
    <div className="fixed inset-0 z-[100] bg-slate-950/60 backdrop-blur-sm flex items-center justify-center p-5">
      <div
        className={`bg-white ${maxWidth} w-full max-h-[90vh] rounded-[24px] shadow-2xl overflow-hidden flex flex-col`}
      >
        <div className="flex items-start justify-between gap-5 px-7 py-5 border-b border-slate-200">
          <div>
            <h3 className="text-xl font-bold">
              {title}
            </h3>

            {subtitle && (
              <p className="text-sm text-slate-500 mt-1">
                {subtitle}
              </p>
            )}
          </div>

          <button
            onClick={
              onClose
            }
            className="w-9 h-9 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-600 flex items-center justify-center transition text-lg"
          >
            ×
          </button>
        </div>

        <div className="overflow-y-auto p-7">
          {children}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const fileRef =
    useRef(null);

  const [
    previewUrl,
    setPreviewUrl,
  ] = useState(null);

  const [
    thumbnailBase64,
    setThumbnailBase64,
  ] = useState(null);

  const [
    uploading,
    setUploading,
  ] = useState(false);

  const [
    material,
    setMaterial,
  ] = useState("");

  const [
    result,
    setResult,
  ] = useState(null);

  const [
    history,
    setHistory,
  ] = useState([]);

  const [
    sidebarOpen,
    setSidebarOpen,
  ] = useState(true);

  const [
    selectedCompareIds,
    setSelectedCompareIds,
  ] = useState([]);

  const [
    showCompareModal,
    setShowCompareModal,
  ] = useState(false);

  const [
    compareSummary,
    setCompareSummary,
  ] = useState(null);

  const [
    compareLoading,
    setCompareLoading,
  ] = useState(false);

  const [
    showAIModal,
    setShowAIModal,
  ] = useState(false);


  const [
    showSimilarModal,
    setShowSimilarModal,
  ] = useState(false);

  const [
    showGradcamModal,
    setShowGradcamModal,
  ] = useState(false);

  const [activeTab, setActiveTab] = useState("fracture");

  const phaseFileRef = useRef(null);

  const [phasePreviewUrl, setPhasePreviewUrl] = useState(null);

  // ==========================================
  // Phase 분석 상태
  // ==========================================

  // 업로드한 원본 이미지 파일
  const [phaseFile, setPhaseFile] = useState(null);

  // Phase 분석 결과
  const [phaseResult, setPhaseResult] = useState(null);

  // Phase 분석 기록
  const [phaseHistory, setPhaseHistory] = useState([]);

  // Phase 분석 진행 상태
  const [phaseLoading, setPhaseLoading] = useState(false);

  // Phase 분석 오류 메시지
  const [phaseError, setPhaseError] = useState("");

// ==========================================
// Phase 이미지 업로드
// ==========================================

const handlePhaseFileChange = async (e) => {

  const file = e.target.files?.[0];

    if (!file) {
      return;
    }

    // 이미지 파일인지 확인
    if (!file.type.startsWith("image/")) {

      alert("이미지 파일만 업로드할 수 있습니다.");

      return;

    }

    try {

      // 이미지 미리보기 생성
      const base64 = await fileToBase64(file);

      // 실제 이미지 파일 저장
      setPhaseFile(file);

      // 이미지 미리보기 저장
      setPhasePreviewUrl(base64);

      // 이전 분석 결과 초기화
      setPhaseResult(null);

      // 이전 오류 초기화
      setPhaseError("");

    } catch (err) {

      console.error(
        "Phase 이미지 처리 실패:",
        err
      );

      alert(
        "이미지를 불러오는 중 오류가 발생했습니다."
      );

    }

  };

  // ==========================================
  // Phase 분석 실행
  // ==========================================

  const handlePhaseAnalyze = async () => {

    // 이미지가 없는 경우
    if (!phaseFile) {

      alert("Phase 분석용 이미지를 업로드해 주세요.");

      return;

    }

    // 중복 실행 방지
    if (phaseLoading) {
      return;
    }

    try {

      setPhaseLoading(true);

      setPhaseError("");

      setPhaseResult(null);

      // 서버로 전달할 데이터
      const formData = new FormData();

      formData.append(
        "file",
        phaseFile
      );

      // Phase 분석 API 호출
      const response = await fetch(
        "http://localhost:8000/analyze-phase",
        {
          method: "POST",
          body: formData,
        }
      );

      // HTTP 오류 처리
      if (!response.ok) {

        throw new Error(
          `Phase 분석 API 오류: ${response.status}`
        );

      }

      // 분석 결과 수신
      const data = await response.json();

if (data.status !== "success") {
  setPhaseError(data.message || "Phase 분석에 실패했습니다.");
  return;
}

setPhaseResult(data);

// 저장용 가벼운 기록 생성 후 Firestore 저장
try {
  const item = await buildPhaseHistoryItem({
    title: `Phase 분석 ${phaseHistory.length + 1}`,
    originalDataUrl: phasePreviewUrl,
    data,
  });
  await addPhaseHistoryItem(item);
  setPhaseHistory(await fetchPhaseHistory());
} catch (err) {
  console.error("Phase 기록 저장 실패:", err);
  // 분석 결과는 이미 화면에 표시됐으므로 저장 실패만 알림
  alert("분석은 완료됐지만 기록 저장에 실패했습니다.");
}

      // 분석 결과 저장
      setPhaseResult(data);

      // Phase 분석 기록 추가
      const newPhaseHistory = {
        id: Date.now(),

        // 자동 제목
        title: `Phase 분석 ${phaseHistory.length + 1}`,

        time: new Date().toLocaleString("ko-KR"),

        image: phasePreviewUrl,

        result: data,
      };

      setPhaseHistory((prev) => [
        newPhaseHistory,
        ...prev,
      ]);

      console.log(
        "Phase 분석 성공:",
        data
      );

    } catch (err) {

      console.error(
        "Phase 분석 실패:",
        err
      );

      setPhaseError(
        "Phase 분석 중 오류가 발생했습니다. 서버 연결 상태를 확인해 주세요."
      );

    } finally {

      setPhaseLoading(false);

    }

  };
 
  // ==========================================
  // Phase 분석 기록 클릭
  // ==========================================

  const handlePhaseHistoryClick = (item) => {
  if (!item) return;
  setPhasePreviewUrl(item.image);   // 썸네일(저해상도)
  setPhaseResult(item.result);
  setPhaseFile(null);               // 이전 파일이 재분석되는 문제 방지
  setPhaseError("");
};

  // ==========================================
  // Phase 기록 제목 변경
  // ==========================================
const handlePhaseHistoryRename = async (id, currentTitle) => {
  const newTitle = window.prompt("새로운 제목을 입력하세요.", currentTitle);
  if (newTitle === null) return;
  const trimmed = newTitle.trim();
  if (!trimmed) return;

  const target = phaseHistory.find((i) => i.id === id);
  setPhaseHistory((prev) =>
    prev.map((i) => (i.id === id ? { ...i, title: trimmed } : i))
  );
  try {
    if (target?.docId) await renamePhaseHistoryItem(target.docId, trimmed);
  } catch (err) {
    console.error("제목 변경 실패:", err);
  }
};

  useEffect(() => {
  const loadHistory = async () => {
    try {
      const items = await fetchHistory();
      setHistory(items);
    } catch (err) {
      console.error("Firestore 기록 불러오기 실패:", err);
    }
  };
  loadHistory();
}, []);

useEffect(() => {
  const loadPhaseHistory = async () => {
    try {
      setPhaseHistory(await fetchPhaseHistory());
    } catch (err) {
      console.error("Phase 기록 불러오기 실패:", err);
    }
  };
  loadPhaseHistory();
}, []);

  const materialText =
    MATERIAL_LABELS[
      result?.material
    ] ||
    result?.material ||
    "-";

  const confidenceStyle =
    CONFIDENCE_BOX_STYLES[
      result
        ?.confidence_status
    ] ||
    CONFIDENCE_BOX_STYLES
      .medium;

  const confidenceLabel =
    result
      ?.confidence_status ===
    "high"
      ? "높음"
      : result
          ?.confidence_status ===
        "low"
      ? "낮음"
      : "보통";

  const compareItems =
    history.filter(
      (item) =>
        selectedCompareIds.includes(
          item.id
        )
    );

  const sameCompareCause =
    compareItems.length >=
      2 &&
    (
      compareItems[0]
        ?.result
        ?.expected_cause ||
      ""
    ).trim() ===
      (
        compareItems[1]
          ?.result
          ?.expected_cause ||
        ""
      ).trim();

  const fileToBase64 = (
    file
  ) => {
    return new Promise(
      (
        resolve,
        reject
      ) => {
        const reader =
          new FileReader();

        reader.onload =
          () =>
            resolve(
              reader.result
            );

        reader.onerror =
          reject;

        reader.readAsDataURL(
          file
        );
      }
    );
  };

  const makeThumbnail = (
    file,
    maxSize = 180,
    quality = 0.65
  ) => {
    return new Promise(
      (
        resolve,
        reject
      ) => {
        const reader =
          new FileReader();

        const img =
          new Image();

        reader.onload =
          () => {
            img.onload =
              () => {
                const canvas =
                  document.createElement(
                    "canvas"
                  );

                const scale =
                  Math.min(
                    maxSize /
                      img.width,
                    maxSize /
                      img.height
                  );

                canvas.width =
                  Math.round(
                    img.width *
                      scale
                  );

                canvas.height =
                  Math.round(
                    img.height *
                      scale
                  );

                const ctx =
                  canvas.getContext(
                    "2d"
                  );

                ctx.drawImage(
                  img,
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );

                resolve(
                  canvas.toDataURL(
                    "image/jpeg",
                    quality
                  )
                );
              };

            img.onerror =
              reject;

            img.src =
              reader.result;
          };

        reader.onerror =
          reject;

        reader.readAsDataURL(
          file
        );
      }
    );
  };

  const handleFileChange =
    async (e) => {
      const file =
        e.target.files[0];

      if (!file) {
        return;
      }

      try {
        const base64 =
          await fileToBase64(
            file
          );

        const thumbnail =
          await makeThumbnail(
            file
          );

        setPreviewUrl(
          base64
        );

        setThumbnailBase64(
          thumbnail
        );

        setResult(null);

        setShowAIModal(
          false
        );

        setShowSimilarModal(
          false
        );

        setShowGradcamModal(
          false
        );
      } catch (err) {
        console.error(
          "이미지 처리 실패:",
          err
        );

        alert(
          "이미지를 불러오는 중 오류가 발생했습니다."
        );
      }
    };

  const saveHistory = async (data, thumbnail) => {
  const hasNewMasks = !!data.gradcam_masks;
  const historyResult = {
    ...data,
    gradcam_image: null,
    gradcam_masks: null,      // Firestore 1MB 제한 고려, 마스크도 제외
    gradcam_layers: null,
  };

  const newItem = {
    id: Date.now(),
    time: new Date().toLocaleString(),
    image: thumbnail,
    result: historyResult,
  };

  try {
    await addHistoryItem(newItem);
    const items = await fetchHistory();
    setHistory(items);
  } catch (err) {
    console.error("Firestore 기록 저장 실패:", err);
    alert("기록 저장 중 오류가 발생했습니다. 콘솔을 확인해주세요.");
  }
};

  const handleHistoryClick =
    (item) => {
      setResult(
        item.result
      );

      setPreviewUrl(
        item.image
      );

      setThumbnailBase64(
        item.image
      );

      setMaterial(
        item.result
          .material ||
          ""
      );

      setShowAIModal(
        false
      );

      setShowSimilarModal(
        false
      );

      setShowGradcamModal(
        false
      );
    };

  const toggleCompareSelect =
    (id) => {
      setSelectedCompareIds(
        (prev) => {
          if (
            prev.includes(
              id
            )
          ) {
            return prev.filter(
              (itemId) =>
                itemId !==
                id
            );
          }

          if (
            prev.length >=
            2
          ) {
            alert(
              "비교는 최대 2개까지 선택할 수 있습니다."
            );

            return prev;
          }

          return [
            ...prev,
            id,
          ];
        }
      );
    };

const clearHistory = async () => {
  try {
    await clearAllHistory();
  } catch (err) {
    console.error("Firestore 기록 삭제 실패:", err);
  }
  setHistory([]);
  setSelectedCompareIds([]);
  setShowCompareModal(false);
  setCompareSummary(null);
};

  const handleUpload =
    async () => {
      const file =
        fileRef.current
          ?.files[0];

      if (!file) {
        return alert(
          "이미지를 먼저 선택해주세요."
        );
      }

      if (!material) {
        return alert(
          "재질을 선택해주세요."
        );
      }

      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      formData.append(
        "material",
        material
      );

      try {
        setUploading(
          true
        );

        const res =
          await fetch(
            "http://localhost:8000/analyze",
            {
              method:
                "POST",
              body:
                formData,
            }
          );

        if (!res.ok) {
          throw new Error(
            `서버 오류: ${res.status}`
          );
        }

        const data =
          await res.json();

        console.log(
          "백엔드 응답:",
          data
        );

        setResult(
          data
        );

        await saveHistory(
          data,
          thumbnailBase64
        );
      } catch (err) {
        console.error(
          "분석 결과 처리 오류:",
          err
        );

        alert(
          "분석 결과 처리 중 오류가 발생했습니다. 콘솔을 확인해주세요."
        );
      } finally {
        setUploading(
          false
        );
      }
    };

  const openCompare =
    () => {
      if (
        selectedCompareIds.length <
        2
      ) {
        alert(
          "왼쪽 분석 기록에서 비교할 결과 2개를 선택해주세요."
        );

        return;
      }

      setCompareSummary(
        null
      );

      setShowCompareModal(
        true
      );
    };

  const handleCompareWithLLM =
    async () => {
      if (
        compareItems.length <
        2
      ) {
        alert(
          "비교할 기록을 2개 선택해주세요."
        );

        return;
      }

      try {
        setCompareLoading(
          true
        );

        setCompareSummary(
          null
        );

        const payload = {
          items:
            compareItems.map(
              (item) =>
                item.result
            ),
        };

        const res =
          await fetch(
            "http://localhost:8000/compare",
            {
              method:
                "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify(
                  payload
                ),
            }
          );

        if (!res.ok) {
          throw new Error(
            `비교 분석 오류: ${res.status}`
          );
        }

        const data =
          await res.json();

        console.log(
          "비교 API 응답:",
          data
        );

        setCompareSummary(
          data
        );
      } catch (err) {
        console.error(
          "LLM 비교 설명 오류:",
          err
        );

        alert(
          "LLM 비교 설명 생성 중 오류가 발생했습니다."
        );
      } finally {
        setCompareLoading(
          false
        );
      }
    };

  return (
    <div
      className={
        layout.page
      }
    >
      <div className="flex min-h-screen">

        {/* 사이드바 열고 닫기 */}
        <button
          onClick={() =>
            setSidebarOpen(
              !sidebarOpen
            )
          }
          className={`fixed top-5 z-50 rounded-r-xl bg-slate-900 text-white px-3 py-3 text-sm shadow-lg transition-all duration-300 ${
            sidebarOpen
              ? "left-80"
              : "left-0"
          }`}
        >
          {sidebarOpen
            ? "‹"
            : "›"}
        </button>

        {/* 분석 기록 사이드바 */}
        <aside
          className={`fixed lg:sticky top-0 left-0 z-40 h-screen bg-white border-r border-slate-200 transition-all duration-300 overflow-y-auto overflow-x-hidden ${
            sidebarOpen
              ? "w-80 translate-x-0 p-5"
              : "w-0 -translate-x-full p-0 border-none"
          }`}
        >
          {activeTab === "fracture" ? (
            <>
              {/* ==========================================
                  파단면 분석 기록
              ========================================== */}

              <div className="flex items-center justify-between mb-5">
                <div>
                  <p className="text-[11px] tracking-[0.16em] uppercase text-slate-400 font-semibold">
                    Fracture History
                  </p>

                  <h2 className="text-lg font-bold mt-1 whitespace-nowrap">
                    파단면 분석 기록
                  </h2>
                </div>

                {history.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
                  >
                    전체 삭제
                  </button>
                )}
              </div>

              {/* 비교 */}
              {history.length > 0 && (
                <div className="mb-4 p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <p className="text-xs text-slate-500 leading-5">
                    결과 2개를 선택하면 비교할 수 있습니다.
                  </p>

                  <button
                    onClick={openCompare}
                    disabled={selectedCompareIds.length < 2}
                    className="mt-2 w-full rounded-lg bg-slate-900 text-white py-2 text-xs font-semibold disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-700 transition"
                  >
                    선택 결과 비교 ({selectedCompareIds.length}/2)
                  </button>
                </div>
              )}

              {/* 파단면 기록 목록 */}
              <div className="space-y-2">
                {history.length === 0 && (
                  <div className="rounded-xl bg-slate-50 border border-dashed p-5 text-center">
                    <p className="text-sm text-slate-400">
                      아직 파단면 분석 기록이 없습니다.
                    </p>
                  </div>
                )}

                {history.map((item) => {
                  const itemResult = item.result;

                  const itemMaterial =
                    MATERIAL_LABELS[itemResult.material] ||
                    itemResult.material ||
                    "-";

                  const checked =
                    selectedCompareIds.includes(item.id);

                  return (
                    <div
                      key={item.id}
                      onClick={() =>
                        handleHistoryClick(item)
                      }
                      className={`relative w-full text-left p-3 rounded-xl border transition cursor-pointer ${
                        checked
                          ? "bg-blue-50 border-blue-400"
                          : "bg-white hover:border-slate-400 hover:bg-slate-50 border-slate-200"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onClick={(e) =>
                          e.stopPropagation()
                        }
                        onChange={() =>
                          toggleCompareSelect(item.id)
                        }
                        className="absolute top-3 right-3 w-4 h-4 accent-slate-900 cursor-pointer"
                      />

                      <div className="flex gap-3 pr-6">
                        <div className="w-14 h-14 rounded-lg bg-slate-200 overflow-hidden shrink-0">
                          {item.image ? (
                            <img
                              src={item.image}
                              alt="기록 이미지"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                              No Image
                            </div>
                          )}
                        </div>

                        <div className="min-w-0">
                          <p className="font-bold text-sm">
                            {itemResult.display_prediction ||
                              itemResult.prediction}
                          </p>

                          <p className="text-sm text-blue-600 font-semibold">
                            {itemResult.confidence}
                          </p>

                          <p className="text-xs text-slate-500 truncate">
                            {itemMaterial}
                          </p>

                          <p className="text-[11px] text-slate-400 mt-1">
                            {item.time}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <>

              {/* ==========================================
                  Phase 분석 기록
              ========================================== */}

              <div className="flex items-center justify-between mb-5">
                <div>
                  <p className="text-[11px] tracking-[0.16em] uppercase text-emerald-600 font-semibold">
                    Phase History
                  </p>

                  <h2 className="text-lg font-bold mt-1 whitespace-nowrap">
                    Phase 분석 기록
                  </h2>
                </div>

                {phaseHistory.length > 0 && (
                  <button
  onClick={async () => {
    try { await clearAllPhaseHistory(); } catch (e) { console.error(e); }
    setPhaseHistory([]);
  }}
  className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
>
  전체 삭제
</button>
                )}
              </div>

              <div className="space-y-2">

                {/* 기록이 없을 때 */}
                {phaseHistory.length === 0 && (
                  <div className="rounded-xl bg-emerald-50/50 border border-dashed border-emerald-200 p-5 text-center">

                    <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-700 font-bold">
                      P
                    </div>

                    <p className="text-sm font-semibold text-slate-600">
                      아직 Phase 분석 기록이 없습니다.
                    </p>

                    <p className="text-xs text-slate-400 mt-2 leading-5">
                      Phase 분석을 실행하면
                      여기에 기록됩니다.
                    </p>

                  </div>
                )}

                {/* Phase 분석 기록 */}
                {phaseHistory.map((item) => {

                return (
                    <div
                      key={item.id}
                      onClick={() =>
                        handlePhaseHistoryClick(item)
                      }
                      className="w-full p-3 rounded-xl border border-slate-200 bg-white hover:border-emerald-400 hover:bg-emerald-50/30 transition cursor-pointer"
                    >

                      <div className="flex gap-3">

                        {/* 원본 이미지 */}
                        <div className="w-14 h-14 rounded-lg bg-slate-200 overflow-hidden shrink-0">

                          {item.image ? (
                            <img
                              src={item.image}
                              alt="Phase 기록 이미지"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                              No Image
                            </div>
                          )}

                        </div>

                        {/* 분석 정보 */}
                        <div className="min-w-0 flex-1">

                          <div className="flex items-start justify-between gap-2">

                            <p className="text-sm font-bold text-slate-700 truncate">
                              {item.title || "Phase 분석"}
                            </p>

                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();

                                handlePhaseHistoryRename(
                                  item.id,
                                  item.title || "Phase 분석"
                                );
                              }}
                              className="shrink-0 text-xs text-slate-400 hover:text-emerald-600 transition"
                              title="제목 변경"
                            >
                              수정
                            </button>

                          </div>
                          
                          <p className="text-[11px] text-slate-400 mt-1">
                            {item.time}
                          </p>

                        </div>

                      </div>

                    </div>
                  );
                })}

              </div>
            </>
          )}
        </aside>

        {/* 메인 */}
        <main className="flex-1 min-w-0">

          {/* 상단 바 */}
        <header className="bg-[#172536] text-white border-b border-slate-700">
          <div
            className={`${layout.container} h-[68px] grid grid-cols-[1fr_auto_1fr] items-center`}
          >
            {/* 로고 */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full border border-white/30 flex items-center justify-center text-lg">
                ◈
              </div>

              <div>
                <h1 className="font-bold text-lg leading-none">
                  Fracture Analysis System
                </h1>

                <p className="text-[10px] text-slate-300 mt-1 tracking-[0.14em] uppercase">
                  AI Fractography Analysis
                </p>
              </div>
            </div>

            {/* 분석 종류 탭 */}
            <div className="flex items-center bg-white/10 p-1 rounded-xl border border-white/10">
              <button
                onClick={() => setActiveTab("fracture")}
                className={`min-w-[150px] px-6 py-2 rounded-lg text-sm font-bold transition ${
                  activeTab === "fracture"
                    ? "bg-blue-600 text-white shadow-md"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`}
              >
                파단면 분석
              </button>

              <button
                onClick={() => setActiveTab("phase")}
                className={`min-w-[150px] px-6 py-2 rounded-lg text-sm font-bold transition ${
                  activeTab === "phase"
                    ? "bg-blue-600 text-white shadow-md"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`}
              >
                Phase 분석
              </button>
            </div>

            <div />
          </div>
        </header>

          <div
            className={`${layout.container} py-8`}
          >
            {activeTab === "fracture" ? (
              <>

            {/* 제목 */}
            <div className="mb-6">
              <p className="text-xs uppercase tracking-[0.18em] text-blue-600 font-bold">
                Failure Analysis
              </p>

              <h2 className="text-2xl font-bold mt-1">
                파손단면 이미지 분석
              </h2>

              <p className="text-sm text-slate-500 mt-2">
                입력 이미지와 모델의
                주요 판단 결과를 한
                화면에서 확인합니다.
              </p>
            </div>

            {/* 메인 분석 카드 */}
            <section className="bg-white rounded-[22px] border border-slate-200 shadow-sm overflow-hidden">

              <div className="grid xl:grid-cols-[1fr_1fr_0.82fr] min-h-[665px]">

                {/* =========================
                    입력 이미지
                ========================== */}
                <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">

                  {/* 제목 */}
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-base">
                      입력 이미지
                    </h3>

                    {previewUrl && (
                      <button
                        onClick={() =>
                          fileRef.current?.click()
                        }
                        className="text-xs text-blue-600 font-semibold hover:text-blue-800"
                      >
                        이미지 변경
                      </button>
                    )}
                  </div>

                  {/* 이미지 + 입력 컨트롤 전체를 세로 중앙 배치 */}
                  <div className="flex-1 flex flex-col justify-center py-5">

                    <label
                      htmlFor="file-input"
                      className="h-[350px] rounded-xl border border-slate-200 bg-slate-50 overflow-hidden flex items-center justify-center cursor-pointer"
                    >
                      {previewUrl ? (
                        <img
                          src={previewUrl}
                          alt="입력 이미지"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="text-center px-5">
                          <div className="w-12 h-12 rounded-full bg-slate-200 mx-auto flex items-center justify-center text-xl mb-3">
                            +
                          </div>

                          <p className="font-semibold text-slate-700">
                            파손단면 이미지를 업로드하세요
                          </p>

                          <p className="text-xs text-slate-400 mt-2">
                            클릭하여 이미지 선택
                          </p>
                        </div>
                      )}
                    </label>

                    <input
                      id="file-input"
                      type="file"
                      accept="image/*"
                      ref={fileRef}
                      onChange={handleFileChange}
                      className="hidden"
                    />

                    {/* 재질 + 분석 버튼 */}
                    <div className="mt-4 flex gap-3">
                      <select
                        value={material}
                        onChange={(e) =>
                          setMaterial(
                            e.target.value
                          )
                        }
                        className="flex-1 h-11 px-3 rounded-lg border border-slate-300 bg-white text-sm outline-none focus:border-blue-500"
                      >
                        <option value="">
                          재질 선택
                        </option>

                        <option value="steel">
                          강 (Steel)
                        </option>

                        <option value="stainless_steel">
                          스테인리스강
                        </option>

                        <option value="aluminum">
                          알루미늄
                        </option>

                        <option value="titanium">
                          티타늄
                        </option>

                        <option value="cast_iron">
                          주철
                        </option>

                        <option value="copper">
                          구리
                        </option>

                        <option value="magnesium">
                          마그네슘 합금
                        </option>

                        <option value="nickel_alloy">
                          니켈 합금
                        </option>

                        <option value="tool_steel">
                          공구강
                        </option>

                        <option value="unknown">
                          모름
                        </option>
                      </select>

                      <button
                        onClick={handleUpload}
                        disabled={uploading}
                        className="px-7 h-11 rounded-lg bg-[#172536] text-white text-sm font-semibold hover:bg-slate-700 disabled:opacity-50 transition"
                      >
                        {uploading
                          ? "분석 중..."
                          : "분석 시작"}
                      </button>
                    </div>
                  </div>
                </div>


                {/* =========================
                    Grad-CAM++
                ========================== */}
                <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">

                  {/* 제목 */}
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-base">
                        Grad-CAM++ 결과
                      </h3>

                      <p className="text-xs text-slate-400 mt-1">
                        모델이 판단에 활용한 영역
                      </p>
                    </div>

                    {result &&
                      (result.gradcam_masks ||
    result.gradcam_layers ||
    (result.gradcam_contours &&
      Object.keys(result.gradcam_contours).length > 0)) && (
                        <button
                          onClick={() =>
                            setShowGradcamModal(
                              true
                            )
                          }
                          className="text-xs px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition"
                        >
                          확대 보기
                        </button>
                      )}
                  </div>

                  {/* Grad-CAM 전체를 세로 중앙 배치 */}
                  <div className="flex-1 flex flex-col justify-center py-5">
                    {result &&
                    (result.gradcam_masks ||
  result.gradcam_layers ||
  (result.gradcam_contours &&
    Object.keys(result.gradcam_contours).length > 0)) ? (
                      <GradcamView
                        result={result}
                        chipSize="text-[11px]"
                        canvasClass="h-[350px]"
                      />
                    ) : (
                      <div className="h-[350px] rounded-xl border border-dashed border-slate-300 bg-slate-50 flex items-center justify-center">
                        <div className="text-center">
                          <p className="text-sm font-semibold text-slate-500">
                            분석 전
                          </p>

                          <p className="text-xs text-slate-400 mt-2">
                            분석을 실행하면 시각화 결과가 표시됩니다.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>


                {/* =========================
                    분석 결과
                ========================== */}
                <div className="p-5 flex flex-col">

                  <h3 className="font-bold text-base">
                    분석 결과
                  </h3>

                  {result ? (
                    <div className="flex-1 flex flex-col pt-4">

                      {/* 최종 예측 */}
                      <div className="rounded-xl bg-gradient-to-br from-blue-50 to-slate-50 border border-blue-100 py-5 px-5 text-center">

                        <p className="text-xs text-slate-500 uppercase tracking-[0.16em]">
                          Predicted fracture
                        </p>

                        <p className="text-3xl font-black text-blue-700 mt-2">
                          {result.display_prediction ||
                            result.prediction}
                        </p>

                        <p className="text-4xl font-black text-slate-900 mt-1">
                          {result.confidence}
                        </p>

                        <span
                          className={`inline-block mt-3 px-3 py-1 rounded-full border text-xs font-bold ${confidenceStyle}`}
                        >
                          신뢰도 {confidenceLabel}
                        </span>
                      </div>


                      {/* 전체 예측 확률 */}
                      {result.similarities && (
                        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">

                          <div className="flex items-center justify-between mb-3">
                            <p className="text-xs font-bold text-slate-600">
                              전체 예측 확률
                            </p>

                            <p className="text-[10px] text-slate-400">
                              예측 확률
                            </p>
                          </div>

                          <div className="space-y-3">
                            {Object.entries(
                              result.similarities
                            )
                              .sort((a, b) => {
  const va = parseFloat(a[1]);
  const vb = parseFloat(b[1]);
  return (isNaN(vb) ? -1 : vb) - (isNaN(va) ? -1 : va);
})
                              .map(
                                ([name, value]) => {
                                  const percent =
                                    Math.max(
                                      0,
                                      Math.min(
                                        100,
                                        parseFloat(
                                          value
                                        ) || 0
                                      )
                                    );

                                  const isPredicted =
                                    name ===
                                      (result.display_prediction ||
                                        result.prediction) ||
                                    EN_NAMES[
                                      name
                                    ] ===
                                      result.prediction_en;

                                  return (
                                    <div key={name}>

                                      <div className="flex items-center justify-between gap-3 mb-1">

                                        <span
                                          className={`text-[11px] ${
                                            isPredicted
                                              ? "font-bold text-slate-900"
                                              : value === "-"
                                              ? "font-medium text-slate-300"
                                              : "font-medium text-slate-500"
                                          }`}
                                        >
                                          {name}
                                        </span>

                                        <span
                                          className={`text-[11px] tabular-nums ${
                                            isPredicted
                                              ? "font-bold text-blue-700"
                                              : "font-semibold text-slate-500"
                                          }`}
                                        >
                                          {typeof value ===
                                          "number"
                                            ? `${value.toFixed(
                                                1
                                              )}%`
                                            : value}
                                        </span>
                                      </div>

                                      {/* 확률 막대 */}
                                      <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                                        <div
                                          className={`h-full rounded-full transition-all duration-500 ${
                                            isPredicted
                                              ? "bg-blue-600"
                                              : "bg-slate-300"
                                          }`}
                                          style={{
                                            width: `${percent}%`,
                                          }}
                                        />
                                      </div>
                                    </div>
                                  );
                                }
                              )}
                          </div>
                        </div>
                      )}


                      {/* 재질 / 상태 */}
                      <div className="mt-4 grid grid-cols-2 gap-3">

                        <div className="rounded-lg border border-slate-200 p-3">
                          <p className="text-[11px] text-slate-400">
                            재질
                          </p>

                          <p className="text-sm font-bold mt-1">
                            {materialText}
                          </p>
                        </div>

                        <div className="rounded-lg border border-slate-200 p-3">
                          <p className="text-[11px] text-slate-400">
                            상태
                          </p>

                          <p className="text-sm font-bold mt-1">
                            {confidenceLabel}
                          </p>
                        </div>
                      </div>


                      {/* 신뢰도 안내 */}
                      <div
                        className={`mt-3 rounded-lg border p-3 ${confidenceStyle}`}
                      >
                        <p className="text-xs leading-5">
                          {result.confidence_message}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="w-full h-[350px] rounded-xl bg-slate-50 border border-dashed border-slate-300 flex items-center justify-center">
                        <div className="text-center px-5">
                          <p className="text-sm font-semibold text-slate-500">
                            분석 결과 없음
                          </p>

                          <p className="text-xs text-slate-400 mt-2">
                            이미지를 선택하고 분석을 시작하세요.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>


              {/* =========================
                  기능 버튼
              ========================== */}
              <div className="border-t border-slate-200 p-5">

                <div className="grid md:grid-cols-2 gap-4">

                  {/* AI 상세 분석 */}
                  <button
                    onClick={() => {
                      if (!result) {
                        alert(
                          "먼저 이미지를 분석해주세요."
                        );

                        return;
                      }

                      setShowAIModal(
                        true
                      );
                    }}
                    className="group rounded-xl bg-blue-600 hover:bg-blue-700 text-white p-4 text-left transition shadow-sm"
                  >
                    <div className="flex items-center gap-3">

                      <div className="w-10 h-10 rounded-lg bg-white/15 flex items-center justify-center font-bold">
                        AI
                      </div>

                      <div>
                        <p className="font-bold">
                          AI 상세 분석
                        </p>

                        <p className="text-xs text-blue-100 mt-0.5">
                          LLM 설명
                        </p>
                      </div>
                    </div>
                  </button>


                  {/* Phase */}
                  


                  {/* 유사 이미지 */}
                  <button
                    onClick={() => {
                      if (!result) {
                        alert(
                          "먼저 이미지를 분석해주세요."
                        );

                        return;
                      }

                      setShowSimilarModal(
                        true
                      );
                    }}
                    className="rounded-xl bg-violet-600 hover:bg-violet-700 text-white p-4 text-left transition shadow-sm"
                  >
                    <div className="flex items-center gap-3">

                      <div className="w-10 h-10 rounded-lg bg-white/15 flex items-center justify-center font-bold">
                        ≋
                      </div>

                      <div>
                        <p className="font-bold">
                          유사 이미지
                        </p>

                        <p className="text-xs text-violet-100 mt-0.5">
                          Top-3 사례
                        </p>
                      </div>
                    </div>
                  </button>
                </div>
              </div>
            </section>

            <div className="mt-5 flex items-start gap-3 rounded-xl border border-slate-200 bg-white px-5 py-4">
              <div className="w-2 h-2 rounded-full bg-blue-600 mt-2 shrink-0" />

              <p className="text-sm text-slate-500 leading-6">
                메인 화면에서는 최종 파손 유형과 전체 예측 확률을
                함께 제공합니다. 상세 특징과 예상 원인은 AI 상세 분석에서,
                유사 사례는 유사 이미지에서 확인할 수 있으며 분석 결과 비교는
                왼쪽 분석 기록에서 두 결과를 선택해 실행할 수 있습니다.
              </p>
            </div>
          
              </>
            ) : (
              <>

              {/* 제목 */}
              <div className="mb-6 flex items-start justify-between gap-6">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-emerald-600 font-bold">
                    Phase Analysis
                  </p>

                  <h2 className="text-2xl font-bold mt-1">
                    Phase 이미지 분석
                  </h2>

                  <p className="text-sm text-slate-500 mt-2">
                    현미경 조직 이미지에서 영역별 Phase를 분류하고 분포를 확인합니다.
                  </p>
                </div>

                <div className="hidden xl:block max-w-[430px] rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-3">
                  <p className="text-xs font-semibold text-emerald-800 leading-5">
                    Phase 분석은 별도의 현미경 조직 이미지를 사용합니다.
                  </p>

                  <p className="text-xs text-slate-500 mt-1 leading-5">
                    파단면 이미지와 Phase 분석용 조직 이미지는 서로 다른 입력입니다.
                  </p>
                </div>
              </div>

              {/* Phase 메인 카드 */}
              <section className="bg-white rounded-[22px] border border-slate-200 shadow-sm overflow-hidden">
                <div className="grid xl:grid-cols-[1fr_1fr_0.82fr] min-h-[560px]">

                  {/* Phase 입력 이미지 */}
                  <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-bold text-base">
                          입력 이미지
                        </h3>

                        <p className="text-xs text-slate-400 mt-1">
                          Phase 분석용 현미경 조직 이미지
                        </p>
                      </div>

                      {phasePreviewUrl && (
                        <button
                          onClick={() =>
                            phaseFileRef.current?.click()
                          }
                          className="text-xs text-emerald-600 font-semibold hover:text-emerald-800"
                        >
                          이미지 변경
                        </button>
                      )}
                    </div>

                    <div className="flex-1 flex flex-col justify-center py-5">
                      <label
                        htmlFor="phase-file-input"
                        className="h-[350px] rounded-xl border border-slate-200 bg-slate-50 overflow-hidden flex items-center justify-center cursor-pointer"
                      >
                        {phasePreviewUrl ? (
                          <img
                            src={phasePreviewUrl}
                            alt="Phase 입력 이미지"
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <div className="text-center px-6">
                            <div className="w-12 h-12 rounded-full bg-emerald-100 text-emerald-700 mx-auto flex items-center justify-center text-xl mb-3">
                              +
                            </div>

                            <p className="font-semibold text-slate-700">
                              현미경 조직 이미지를 업로드하세요
                            </p>

                            <p className="text-xs text-slate-400 mt-2">
                              클릭하여 Phase 분석용 이미지 선택
                            </p>
                          </div>
                        )}
                      </label>

                      <input
                        id="phase-file-input"
                        type="file"
                        accept="image/*"
                        ref={phaseFileRef}
                        onChange={handlePhaseFileChange}
                        className="hidden"
                      />

                      <button
                        onClick={handlePhaseAnalyze}
                        disabled={!phaseFile || phaseLoading}
                        className="mt-4 w-full h-11 rounded-lg bg-[#172536] text-white text-sm font-semibold hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
                      >
                        {phaseLoading
                          ? "Phase 분석 중..."
                          : "Phase 분석 시작"}
                      </button>
                    </div>
                  </div>

                  {/* ==========================================
                      Phase Segmentation 결과
                  ========================================== */}

                  <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">

                    <div>
                      <h3 className="font-bold text-base">
                        Phase Segmentation 결과
                      </h3>

                      <p className="text-xs text-slate-400 mt-1">
                        AI가 분류한 영역별 Phase 시각화
                      </p>
                    </div>

                    <div className="flex-1 flex flex-col justify-center py-5">

                      <div className="w-full h-[350px] rounded-xl border border-slate-200 bg-slate-50 overflow-hidden flex items-center justify-center">

                        {phaseLoading ? (

                          <div className="text-center">

                            <div className="w-9 h-9 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />

                            <p className="text-sm text-slate-500">
                              Phase 분석 중...
                            </p>

                          </div>

                        ) : phaseResult?.overlay_image ? (

                          <img
                            src={phaseResult.overlay_image}
                            alt="Phase Segmentation 결과"
                            className="w-full h-full object-contain"
                          />

                        ) : (

                          <div className="text-center px-6">

                            <div className="w-14 h-14 rounded-full bg-emerald-100 text-emerald-700 mx-auto flex items-center justify-center font-bold text-xl">
                              P
                            </div>

                            <p className="text-sm font-semibold text-slate-600 mt-4">
                              Segmentation 결과 없음
                            </p>

                            <p className="text-xs text-slate-400 mt-2 leading-5">
                              Phase 이미지를 업로드하고 분석을 시작하세요.
                            </p>

                          </div>

                        )}

                      </div>

                      {/* 분석 오류 */}

                      {phaseError && (

                        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3">

                          <p className="text-xs text-red-600">
                            {phaseError}
                          </p>

                        </div>

                      )}

                    </div>

                  </div>

                  {/* Phase 분석 결과 */}
                  <div className="p-5 flex flex-col">
                    <div>
                      <h3 className="font-bold text-base">
                        Phase 분석 결과
                      </h3>

                      <p className="text-xs text-slate-400 mt-1">
                        Phase별 면적 구성 비율
                      </p>
                    </div>

                    <div className="flex-1 flex items-center justify-center py-5">
                      <div className="w-full">
                        <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                          <p className="text-xs font-bold text-slate-600">
                            Phase 구성 비율
                          </p>

                          <div className="mt-5 space-y-5">
                            {[
                              {
                                label: "Primary Si",
                                key: "Primary Si",
                              },
                              {
                                label: "Eutectic Si",
                                key: "Eutectic Si",
                              },
                              {
                                label: "Al₃Ni",
                                key: "Al3Ni",
                              },
                              {
                                label: "Al",
                                key: "Al",
                              },
                            ].map((phase) => {

                              const value =
                                phaseResult?.phase_distribution?.[phase.key];

                              return (

                                <div key={phase.key}>

                                  <div className="flex items-center justify-between mb-2">

                                    <span className="text-xs font-medium text-slate-500">
                                      {phase.label}
                                    </span>

                                    <span className="text-xs font-semibold text-slate-700">
                                      {typeof value === "number"
                                        ? `${value.toFixed(1)}%`
                                        : "-"}
                                    </span>

                                  </div>

                                  <div className="h-2 rounded-full bg-slate-200 overflow-hidden">

                                    <div
                                      className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                                      style={{
                                        width: `${
                                          typeof value === "number"
                                            ? Math.min(100, Math.max(0, value))
                                            : 0
                                        }%`
                                      }}
                                    />

                                  </div>

                                </div>

                              );

                            })}

                          </div>
                        </div>

                        <div className="mt-4 rounded-xl border border-emerald-100 bg-emerald-50 p-4">
                          <div className="flex gap-3">
                            <div className="w-5 h-5 rounded-full bg-emerald-600 text-white flex items-center justify-center text-[10px] font-bold shrink-0">
                              i
                            </div>

                            <p className="text-xs text-emerald-700 leading-5">
                              모델 연결 후 segmentation 결과를 기준으로 각 Phase의 면적 비율이 표시됩니다.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              <div className="mt-5 flex items-start gap-3 rounded-xl border border-slate-200 bg-white px-5 py-4">
                <div className="w-2 h-2 rounded-full bg-emerald-600 mt-2 shrink-0" />

                <p className="text-sm text-slate-500 leading-6">
                  Phase 분석은 파단면 분석과 별도의 입력 이미지와 모델을 사용합니다.
                  분석 결과에는 영역별 Phase 분류 이미지와 각 Phase의 면적 비율이 표시됩니다.
                  면적 비율은 이미지에서 각 Phase가 차지하는 픽셀 비율이며,
                  모델의 예측 신뢰도와는 다른 값입니다.
                </p>
              </div>

              </>
            )}
          </div>
        </main>
      </div>

      {/* AI 상세 분석 */}
      {showAIModal &&
        result && (
          <ModalShell
            title="AI 상세 분석"
            subtitle="Gemma 기반 파손단면 분석 설명"
            onClose={() =>
              setShowAIModal(
                false
              )
            }
            maxWidth="max-w-4xl"
          >
            <div className="space-y-4">

              <div className="rounded-xl border border-slate-200 p-5">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold">
                    1
                  </span>

                  <h4 className="font-bold">
                    주요 특징
                  </h4>
                </div>

                <p className="text-sm text-slate-700 leading-7">
                  {result.feature ||
                    "주요 특징 정보가 없습니다."}
                </p>
              </div>

              <div className="rounded-xl border border-slate-200 p-5">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-full bg-indigo-600 text-white flex items-center justify-center text-xs font-bold">
                    2
                  </span>

                  <h4 className="font-bold">
                    판단 근거 설명
                  </h4>
                </div>

                <p className="text-sm text-slate-700 leading-7 whitespace-pre-line">
                  {result.explanation ||
                    "설명 정보가 없습니다."}
                </p>
              </div>

              <div className="rounded-xl border border-slate-200 p-5">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-full bg-amber-500 text-white flex items-center justify-center text-xs font-bold">
                    3
                  </span>

                  <h4 className="font-bold">
                    예상 원인
                  </h4>
                </div>

                <p className="text-sm text-slate-700 leading-7">
                  {result.expected_cause ||
                    "예상 원인 정보가 없습니다."}
                </p>
              </div>

              <div
                className={`rounded-xl border p-5 ${confidenceStyle}`}
              >
                <p className="font-bold text-sm">
                  분석 신뢰도 안내
                </p>

                <p className="text-sm leading-6 mt-2">
                  {
                    result.confidence_message
                  }
                </p>
              </div>

              <p className="text-xs text-slate-400 leading-5">
                본 설명은 이미지 및 입력
                정보를 기반으로 생성된 AI
                분석 결과이며 실제 사고
                원인을 확정하는 정보가
                아닙니다.
              </p>
            </div>
          </ModalShell>
        )}

      {/* 유사 이미지 */}
      {showSimilarModal &&
        result && (
          <ModalShell
            title="유사 이미지"
            subtitle="입력 이미지와 특징이 유사한 사례 Top-3"
            onClose={() =>
              setShowSimilarModal(
                false
              )
            }
            maxWidth="max-w-5xl"
          >
            {result
              ?.similar_images
              ?.length >
            0 ? (
              <div className="grid md:grid-cols-3 gap-5">
                {result.similar_images
                  .slice(
                    0,
                    3
                  )
                  .map(
                    (
                      item,
                      index
                    ) => (
                      <div
                        key={
                          index
                        }
                        className="rounded-xl border border-slate-200 overflow-hidden bg-white"
                      >
                        <div className="h-52 bg-slate-100">
                          <img
                            src={`http://localhost:8000${item.image_url}`}
                            alt={`유사 사례 ${
                              index +
                              1
                            }`}
                            className="w-full h-full object-cover"
                          />
                        </div>

                        <div className="p-4">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-bold text-violet-600">
                              TOP{" "}
                              {index +
                                1}
                            </span>

                            {item.similarity !==
                              undefined && (
                              <span className="text-sm font-bold">
                                {typeof item.similarity ===
                                "number"
                                  ? `${(
                                      item.similarity *
                                      100
                                    ).toFixed(
                                      1
                                    )}%`
                                  : item.similarity}
                              </span>
                            )}
                          </div>

                          <p className="font-bold mt-2">
                            {item.class_name ||
                              item.label ||
                              result.display_prediction ||
                              result.prediction}
                          </p>

                          {item.filename && (
                            <p className="text-xs text-slate-400 mt-1">
                              {
                                item.filename
                              }
                            </p>
                          )}
                        </div>
                      </div>
                    )
                  )}
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 py-20 text-center">
                <p className="font-semibold text-slate-600">
                  유사 이미지 결과가
                  없습니다.
                </p>

                <p className="text-sm text-slate-400 mt-2">
                  백엔드 응답의
                  similar_images 결과를
                  확인해주세요.
                </p>
              </div>
            )}
          </ModalShell>
        )}

      {/* GradCAM 확대 */}
      {showGradcamModal &&
        result && (
          <ModalShell
            title="Grad-CAM++ 확대 보기"
            subtitle="모델 활성화 영역을 확대하여 확인합니다."
            onClose={() =>
              setShowGradcamModal(
                false
              )
            }
            maxWidth="max-w-5xl"
          >
            <GradcamView
              result={
                result
              }
              chipSize="text-sm"
              canvasClass="max-h-[70vh] min-h-[500px]"
            />
          </ModalShell>
        )}

      {/* 비교 모달 */}
      {showCompareModal && (
        <ModalShell
          title="분석 결과 비교"
          subtitle="선택한 두 분석 결과의 특징과 차이를 비교합니다."
          onClose={() =>
            setShowCompareModal(
              false
            )
          }
          maxWidth="max-w-6xl"
        >
          {compareItems.length <
          2 ? (
            <div className="rounded-xl bg-slate-50 border p-10 text-center text-slate-500">
              비교할 기록 2개를
              선택해주세요.
            </div>
          ) : (
            <>
              <div className="grid md:grid-cols-2 gap-5">
                {compareItems.map(
                  (
                    item,
                    index
                  ) => {
                    const itemResult =
                      item.result;

                    const itemMaterial =
                      MATERIAL_LABELS[
                        itemResult
                          .material
                      ] ||
                      itemResult
                        .material ||
                      "-";

                    return (
                      <div
                        key={
                          item.id
                        }
                        className="rounded-xl border border-slate-200 overflow-hidden"
                      >
                        <div className="bg-slate-50 px-5 py-3 border-b flex items-center justify-between">
                          <p className="font-bold">
                            이미지{" "}
                            {index +
                              1}
                          </p>

                          <p className="text-xs text-slate-400">
                            {
                              item.time
                            }
                          </p>
                        </div>

                        <div className="p-5">
                          <div className="h-56 rounded-xl border bg-white overflow-hidden flex items-center justify-center">
                            {item.image ? (
                              <img
                                src={
                                  item.image
                                }
                                alt="비교 이미지"
                                className="w-full h-full object-contain"
                              />
                            ) : (
                              <p className="text-slate-400">
                                No Image
                              </p>
                            )}
                          </div>

                          <div className="mt-4 flex items-end justify-between">
                            <div>
                              <p className="text-xs text-slate-400">
                                파손 유형
                              </p>

                              <p className="text-xl font-bold mt-1">
                                {itemResult
                                  .display_prediction ||
                                  itemResult
                                    .prediction}
                              </p>
                            </div>

                            <p className="text-2xl font-black text-blue-600">
                              {
                                itemResult
                                  .confidence
                              }
                            </p>
                          </div>

                          <div className="mt-4 rounded-lg bg-slate-50 p-4">
                            <p className="text-xs text-slate-400">
                              재질
                            </p>

                            <p className="text-sm font-semibold mt-1">
                              {
                                itemMaterial
                              }
                            </p>
                          </div>

                          <div className="mt-3 rounded-lg border p-4">
                            <p className="text-xs font-bold text-slate-500">
                              주요 특징
                            </p>

                            <p className="text-sm leading-6 mt-2">
                              {itemResult.feature ||
                                "-"}
                            </p>
                          </div>

                          <div className="mt-3 rounded-lg border p-4">
                            <p className="text-xs font-bold text-slate-500">
                              예상 원인
                            </p>

                            <p className="text-sm leading-6 mt-2">
                              {itemResult.expected_cause ||
                                "-"}
                            </p>
                          </div>
                        </div>
                      </div>
                    );
                  }
                )}
              </div>

              <div className="mt-6 flex justify-center">
                <button
                  onClick={
                    handleCompareWithLLM
                  }
                  disabled={
                    compareLoading
                  }
                  className="rounded-xl bg-[#172536] text-white px-7 py-3 text-sm font-bold hover:bg-slate-700 disabled:opacity-50 transition"
                >
                  {compareLoading
                    ? "비교 설명 생성 중..."
                    : "AI 비교 설명 생성"}
                </button>
              </div>

              {compareSummary && (
                <div className="mt-7 space-y-4">

                  <div className="rounded-xl border border-blue-200 bg-blue-50 p-5">
                    <p className="text-xs font-bold text-blue-600">
                      핵심 요약
                    </p>

                    <p className="text-lg font-bold text-blue-950 leading-8 mt-2">
                      {compareSummary.summary ||
                        compareSummary.compare_summary ||
                        "비교 요약이 없습니다."}
                    </p>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">

                    <div className="rounded-xl border p-5">
                      <p className="font-bold">
                        공통점
                      </p>

                      <p className="text-sm text-slate-600 leading-7 mt-2">
                        {compareSummary.common_point ||
                          "공통점 정보가 없습니다."}
                      </p>
                    </div>

                    <div className="rounded-xl border p-5">
                      <p className="font-bold">
                        시각적 특징 차이
                      </p>

                      <p className="text-sm text-slate-600 leading-7 mt-2">
                        {compareSummary.visual_difference ||
                          "시각적 차이 정보가 없습니다."}
                      </p>
                    </div>
                  </div>

                  <div className="rounded-xl border p-5">
                    <p className="font-bold">
                      파손 메커니즘 차이
                    </p>

                    <p className="text-sm text-slate-600 leading-7 mt-2">
                      {compareSummary.mechanism_difference ||
                        "메커니즘 차이 정보가 없습니다."}
                    </p>
                  </div>

                  {sameCompareCause ? (
                    <div className="rounded-xl border border-rose-200 bg-rose-50 p-5">
                      <p className="font-bold text-rose-800">
                        공통 예상 원인
                      </p>

                      <p className="text-sm text-rose-700 leading-7 mt-2">
                        {compareItems[0]
                          ?.result
                          ?.expected_cause ||
                          "-"}
                      </p>
                    </div>
                  ) : (
                    <div className="rounded-xl border p-5">
                      <p className="font-bold">
                        예상 원인 차이
                      </p>

                      <p className="text-sm text-slate-600 leading-7 mt-2">
                        {compareSummary.cause_difference ||
                          "예상 원인 차이 정보가 없습니다."}
                      </p>
                    </div>
                  )}

                  <div className="rounded-xl border p-5">
                    <p className="font-bold">
                      예측 확률 비교
                    </p>

                    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-5 mt-4">

                      <div>
                        <p className="text-xs text-slate-400">
                          이미지 1
                        </p>

                        <p className="text-2xl font-black mt-1">
                          {compareItems[
                            0
                          ]?.result
                            ?.confidence ||
                            "-"}
                        </p>
                      </div>

                      <div className="text-slate-300 font-black">
                        VS
                      </div>

                      <div className="text-right">
                        <p className="text-xs text-slate-400">
                          이미지 2
                        </p>

                        <p className="text-2xl font-black mt-1">
                          {compareItems[
                            1
                          ]?.result
                            ?.confidence ||
                            "-"}
                        </p>
                      </div>
                    </div>

                    <p className="text-sm text-slate-600 leading-7 mt-4">
                      {compareSummary.confidence_difference ||
                        "예측 확률 비교 정보가 없습니다."}
                    </p>
                  </div>

                  <p className="text-xs text-slate-400 leading-5">
                    본 비교 분석은 이미지와
                    입력 정보를 기반으로 한
                    AI 추정 결과입니다.
                  </p>
                </div>
              )}
            </>
          )}
        </ModalShell>
      )}
    </div>
  );
}
