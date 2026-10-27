"""
RAG 검색 모듈 (파단면 분석용)

흐름
  1) backend/rag_docs/ 의 PDF / TXT / MD 문서를 읽어 청크로 분할
  2) Ollama 임베딩 모델(bge-m3)로 벡터화해서 backend/rag_index/ 에 저장
  3) CNN이 예측한 파손 유형으로 질의를 만들고, 코사인 유사도로 상위 청크 검색
  4) 검색 결과를 LLM 프롬프트의 [참고 문헌] 섹션과 API 응답의 references 로 전달

설계 메모
  - 코퍼스가 논문 몇 편 규모라 벡터 DB 없이 numpy 행렬 하나로 충분하다.
    (추가 의존성 없음. 문서가 수만 청크로 늘면 FAISS/Chroma 로 교체)
  - 인덱스/검색 어느 단계가 실패해도 /analyze 가 죽지 않도록
    retrieve_references() 는 항상 빈 리스트로 폴백한다. (llm_service 의 폴백 방식과 동일)
  - 재질은 질의에 넣지 않는다. llm_service 프롬프트가
    "재질을 판단 근거로 쓰지 않는다"는 규칙을 이미 갖고 있기 때문이다.
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np


# =========================================================
# 설정
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAG_DOCS_DIR = os.path.join(BASE_DIR, "rag_docs")
RAG_INDEX_DIR = os.path.join(BASE_DIR, "rag_index")

# 한/영 혼용 검색에 강한 다국어 임베딩 모델: `ollama pull bge-m3`
EMBED_MODEL = "bge-m3"

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
EMBED_BATCH = 16

TOP_K = 4

# bge-m3 코사인 유사도 기준. 실제 문서로 돌려 보고 조정할 것.
MIN_SCORE = 0.35

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")

# 인덱싱에서 제외할 파일 (안내용 문서)
IGNORED_FILENAMES = {"readme.md"}


# =========================================================
# 파손 유형별 검색 질의 (코퍼스가 영문 논문이므로 영어)
# =========================================================

CLASS_QUERIES = {
    "Cleavage": (
        "brittle cleavage fracture facets river patterns "
        "causes low temperature impact loading stress concentration"
    ),
    "Ductile": (
        "ductile overload fracture microvoid coalescence dimples "
        "plastic deformation causes"
    ),
    "Fatigue": (
        "fatigue fracture beach marks striations cyclic loading "
        "crack initiation propagation causes"
    ),
    "Intergranular": (
        "intergranular fracture grain boundary embrittlement "
        "hydrogen stress corrosion cracking segregation precipitates causes"
    ),
}


def build_query(predicted_class_en: str) -> str:
    return CLASS_QUERIES.get(
        predicted_class_en,
        f"{predicted_class_en} fracture surface mechanism causes",
    )


# =========================================================
# 문서 읽기 / 청킹
# =========================================================

def _clean(text: str) -> str:
    # PDF 합자(ﬁ, ﬂ 등)와 첨자를 일반 문자로 정규화 (identiﬁed → identified)
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\r", "\n")

    # 줄바꿈으로 끊긴 하이픈 단어 결합 (fracture-\nsurface → fracturesurface 방지를 위해 소문자 이어짐만)
    text = re.sub(r"-\n(?=[a-z])", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _drop_references(pages: List[Tuple[Optional[int], str]]):
    """
    논문 뒷부분의 참고문헌 목록은 검색 노이즈이므로 제거한다.
    문서 뒤쪽 40% 안에서 'References' 단독 줄이 나오면 그 지점부터 자른다.
    """
    if len(pages) < 3:
        return pages

    start_page = int(len(pages) * 0.6)

    for idx in range(start_page, len(pages)):
        page_no, text = pages[idx]

        match = re.search(
            r"(?im)^\s*(?:\d+\.\s*)?references\s*$",
            text,
        )

        if match:
            trimmed = text[: match.start()].strip()
            kept = pages[:idx]

            if trimmed:
                kept.append((page_no, trimmed))

            return kept

    return pages


def extract_pages(path: str) -> List[Tuple[Optional[int], str]]:
    """(페이지 번호, 텍스트) 목록. TXT/MD 는 페이지 번호 None."""

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)

        pages = []

        for i, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""

            pages.append((i, _clean(raw)))

        return _drop_references(pages)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [(None, _clean(f.read()))]


def chunk_text(
    text: str,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """문장 경계를 최대한 지키면서 size 글자 안팎으로 분할."""

    # 문단 안의 줄바꿈은 공백으로, 공백은 하나로
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。])\s+", text)

    chunks: List[str] = []
    cur = ""

    for sentence in sentences:

        # 문장 하나가 size 보다 길면 강제 분할
        if len(sentence) > size:

            if cur:
                chunks.append(cur)
                cur = ""

            step = max(size - overlap, 1)

            for i in range(0, len(sentence), step):
                chunks.append(sentence[i : i + size])

            continue

        if cur and len(cur) + 1 + len(sentence) > size:

            chunks.append(cur)

            tail = cur[-overlap:] if overlap else ""

            # 겹침 구간이 단어 중간에서 시작하지 않도록 첫 공백 뒤부터 사용
            if " " in tail:
                tail = tail[tail.find(" ") + 1 :]

            cur = (tail + " " + sentence).strip()

        else:
            cur = (cur + " " + sentence).strip() if cur else sentence

    if cur:
        chunks.append(cur)

    # 너무 짧은 조각(제목 파편 등)은 버린다
    return [c for c in chunks if len(c) >= 40]


# =========================================================
# 임베딩 (Ollama)
# =========================================================

def embed_texts(texts: List[str]) -> np.ndarray:
    """텍스트 목록 → L2 정규화된 float32 행렬 (N, D)."""

    import ollama

    vectors = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]

        response = ollama.embed(
            model=EMBED_MODEL,
            input=batch,
        )

        vectors.extend(response["embeddings"])

    arr = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)

    return arr / np.clip(norms, 1e-12, None)


# =========================================================
# 인덱스
# =========================================================

class RagIndex:

    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = index_dir or RAG_INDEX_DIR
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return (
            self.embeddings is not None
            and len(self.chunks) > 0
        )

    # ---------------------------------------------------
    # 빌드
    # ---------------------------------------------------

    def build(self, docs_dir: Optional[str] = None):
        """docs_dir 의 문서를 읽어 인덱스를 만들고 저장한다. (문서 수, 청크 수) 반환."""

        docs_dir = docs_dir or RAG_DOCS_DIR

        os.makedirs(docs_dir, exist_ok=True)

        files = sorted(
            f
            for f in os.listdir(docs_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
            and f.lower() not in IGNORED_FILENAMES
        )

        chunks: List[Dict] = []

        for fname in files:

            path = os.path.join(docs_dir, fname)

            before = len(chunks)

            for page_no, text in extract_pages(path):
                for piece in chunk_text(text):
                    chunks.append(
                        {
                            "source": fname,
                            "page": page_no,
                            "text": piece,
                        }
                    )

            print(f"[RAG] {fname}: {len(chunks) - before} chunks")

        if not chunks:
            raise ValueError(
                f"인덱싱할 문서가 없습니다: {docs_dir} "
                f"({', '.join(SUPPORTED_EXTENSIONS)} 파일을 넣어 주세요)"
            )

        embeddings = embed_texts([c["text"] for c in chunks])

        os.makedirs(self.index_dir, exist_ok=True)

        np.save(
            os.path.join(self.index_dir, "embeddings.npy"),
            embeddings,
        )

        with open(
            os.path.join(self.index_dir, "chunks.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "embed_model": EMBED_MODEL,
                    "chunks": chunks,
                },
                f,
                ensure_ascii=False,
            )

        self.chunks = chunks
        self.embeddings = embeddings

        return len(files), len(chunks)

    # ---------------------------------------------------
    # 로드
    # ---------------------------------------------------

    def load(self) -> bool:
        emb_path = os.path.join(self.index_dir, "embeddings.npy")
        meta_path = os.path.join(self.index_dir, "chunks.json")

        if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
            return False

        embeddings = np.load(emb_path, allow_pickle=False)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        chunks = meta.get("chunks", [])

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "rag_index 가 손상되었습니다 (청크 수와 벡터 수 불일치). "
                "build_rag_index.py 를 다시 실행하세요."
            )

        if meta.get("embed_model") != EMBED_MODEL:
            print(
                f"[RAG] 경고: 인덱스는 '{meta.get('embed_model')}' 로 만들어졌고 "
                f"현재 설정은 '{EMBED_MODEL}' 입니다. 인덱스를 다시 빌드하세요."
            )

        self.chunks = chunks
        self.embeddings = embeddings

        return True

    # ---------------------------------------------------
    # 검색
    # ---------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> List[Dict]:

        if not self.ready:
            return []

        q = embed_texts([query])[0]

        if q.shape[0] != self.embeddings.shape[1]:
            print(
                "[RAG] 임베딩 차원이 인덱스와 다릅니다. 인덱스를 다시 빌드하세요."
            )
            return []

        scores = self.embeddings @ q

        order = np.argsort(-scores)

        results: List[Dict] = []

        # 같은 문단이 거의 그대로 반복되는 청크(중복 페이지, 합자 차이 등)는 한 번만 사용
        seen_keys = set()

        for i in order:

            score = float(scores[i])

            if score < min_score:
                break

            chunk = self.chunks[int(i)]

            key = re.sub(r"[^a-z0-9]", "", chunk["text"].lower())[:150]

            if key in seen_keys:
                continue

            seen_keys.add(key)

            results.append(
                {
                    "source": chunk["source"],
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                    "score": round(score, 4),
                }
            )

            if len(results) >= top_k:
                break

        return results


# =========================================================
# main.py / llm_service.py 에서 쓰는 진입점
# =========================================================

_index: Optional[RagIndex] = None


def get_index() -> RagIndex:
    """인덱스를 지연 로드한다. 아직 없으면 다음 호출에서 다시 시도한다."""

    global _index

    if _index is None or not _index.ready:
        idx = RagIndex()

        try:
            idx.load()
        except Exception as e:
            print(f"[RAG] 인덱스 로드 실패: {e}")

        _index = idx

    return _index


def retrieve_references(
    predicted_class_en: str,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    예측 유형에 맞는 문헌 청크를 검색한다.
    인덱스가 없거나 Ollama 오류가 나도 빈 리스트를 돌려주어 분석은 계속된다.
    """

    try:
        idx = get_index()

        if not idx.ready:
            return []

        refs = idx.search(
            build_query(predicted_class_en),
            top_k=top_k,
        )

        print(
            f"[RAG] {predicted_class_en}: {len(refs)}개 청크 검색"
        )

        return refs

    except Exception as e:
        print(f"[RAG] 검색 실패 → 참고 문헌 없이 진행: {e}")
        return []


def format_references_for_prompt(
    refs: List[Dict],
    max_chars: int = 700,
) -> str:
    """검색 결과를 LLM 프롬프트에 넣을 텍스트로 변환."""

    if not refs:
        return ""

    blocks = []

    for n, ref in enumerate(refs, start=1):

        location = ref["source"]

        if ref.get("page"):
            location += f", p.{ref['page']}"

        blocks.append(
            f"[{n}] ({location})\n{ref['text'][:max_chars]}"
        )

    return "\n\n".join(blocks)


def public_references(
    refs: List[Dict],
    snippet_chars: int = 240,
) -> List[Dict]:
    """프론트에 내려줄 출처 정보 (본문은 짧은 발췌만)."""

    return [
        {
            "source": ref["source"],
            "page": ref.get("page"),
            "score": ref["score"],
            "snippet": ref["text"][:snippet_chars],
        }
        for ref in refs
    ]
