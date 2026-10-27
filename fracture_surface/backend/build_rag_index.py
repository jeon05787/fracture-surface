"""
RAG 인덱스 빌드 스크립트

사용법 (backend 폴더에서, Ollama 실행 중 + `ollama pull bge-m3` 완료 상태):

    python build_rag_index.py
    python build_rag_index.py --docs 다른_문서_폴더

backend/rag_docs/ 에 PDF / TXT / MD 문서를 넣고 실행하면
backend/rag_index/ 에 embeddings.npy, chunks.json 이 생성된다.
문서를 추가하거나 바꿨다면 다시 실행해야 한다.
"""

import argparse
import sys

import rag_service


def main() -> int:

    parser = argparse.ArgumentParser(
        description="RAG 인덱스 빌드"
    )

    parser.add_argument(
        "--docs",
        default=None,
        help=f"문서 폴더 (기본값: {rag_service.RAG_DOCS_DIR})",
    )

    args = parser.parse_args()

    index = rag_service.RagIndex()

    try:
        n_files, n_chunks = index.build(args.docs)

    except ValueError as e:
        print(f"[오류] {e}")
        return 1

    except Exception as e:
        print(
            f"[오류] 인덱스 생성 실패: {e}\n"
            "Ollama 가 실행 중인지(ollama serve), "
            f"임베딩 모델이 설치됐는지(ollama pull {rag_service.EMBED_MODEL}) 확인하세요."
        )
        return 1

    print(
        f"완료: 문서 {n_files}개 / 청크 {n_chunks}개 → {index.index_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
