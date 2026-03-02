# FRAMES RAG System

[FRAMES Benchmark](https://huggingface.co/datasets/google/frames-benchmark) 기반 Multi-hop QA를 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 성능

| Version | Model | Accuracy | Key Changes |
|---------|-------|----------|-------------|
| V1 | gpt-4o-mini | 31% | Baseline (wiki 미사용) |
| V3 | gpt-4o-mini | 51% | top_k=8, context=6000, CoT 프롬프트 |
| V9 | gpt-4o-mini | 57% | Blended reranking (bi 60% + cross 40%) |
| V11 | gpt-4o-mini | 61% | Source-balanced retrieval + context 확대 |
| **V12b** | **gpt-5-mini** | **65%** | Anti-refusal + arithmetic check 프롬프트 |
| V12b | gpt-4o-mini | 64% | V12b 프롬프트 적용 (안정적) |

## 아키텍처

```
Wikipedia URL 파싱
  → Wikipedia 콘텐츠 Fetch (비동기, 캐싱)
  → Document Chunking (문장 기반, 450토큰 목표, 25% 오버랩)
  → Embedding (BAAI/bge-small-en-v1.5, 로컬)
  → Semantic Retrieval (코사인 유사도, 3배수 over-fetch)
  → Blended Reranking (bi-encoder 60% + cross-encoder 40%)
  → Source-balanced Selection (소스별 최소 1청크 보장)
  → LLM Generation (Chain-of-Thought prompting)
  → Evaluation (GPT-5-mini, LLM-as-a-Judge)
```

## 프로젝트 구조

```
├── run.py                  # 메인 RAG 파이프라인 (V12b)
├── wikipedia_fetcher.py    # Wikipedia 콘텐츠 비동기 fetch + 캐싱
├── chunker.py              # 문장 기반 문서 청킹 (오버랩 지원)
├── embedder.py             # 임베딩 생성 (sentence-transformers, 캐싱)
├── retriever.py            # 시맨틱 검색 (코사인 유사도)
├── prompts.py              # CoT 프롬프트 템플릿
├── query_decomposer.py     # 질문 분해 (V7, 비활성화)
├── DEVELOPMENT_LOG.md      # 전체 실험 이력 (V1~V13)
├── final/                  # 최종 제출 코드
└── evaluation_results_*.json  # 실험 결과 파일
```

## 실행 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

`.env` 파일에 API 키 설정:
```
OPENAI_API_KEY=sk-...
HF_ACCESS_TOKEN=hf_...
```

### 실행

```bash
# 기본 실행 (gpt-4o-mini, 0~99 범위)
python run.py --model gpt-4o-mini --start 0 --end 100

# 병렬 처리 (5 workers)
python run.py --model gpt-4o-mini --start 0 --end 100 --workers 5

# Batch API 모드 (50% 비용 절감)
python run.py --model gpt-4o-mini --start 0 --end 100 --batch

# 버전 지정
python run.py --model gpt-4o-mini --start 100 --end 200 --version v11
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 생성 모델 (gpt-4o-mini, gpt-5-mini) | gpt-4o-mini |
| `--start` | 시작 인덱스 | 0 |
| `--end` | 종료 인덱스 | 100 |
| `--workers` | 병렬 워커 수 | 1 |
| `--batch` | OpenAI Batch API 사용 | false |
| `--version` | 결과 파일 버전 접미사 | "" |
| `--verbose` | 상세 로그 출력 | false |

## 주요 설계 결정

### Blended Reranking (V9, +4%)
bi-encoder(bge-small)로 후보 30개를 추출한 뒤 cross-encoder(ms-marco-MiniLM)로 재평가합니다. cross-encoder 점수만 사용하면 수치/표 데이터 청크가 밀려나는 현상이 관찰되어, 두 점수를 min-max 정규화 후 6:4 비율로 블렌딩합니다.

### Source-balanced Selection (V11, +4%)
Multi-hop QA에서는 여러 위키 문서의 정보를 조합해야 합니다. relevance score만으로 선택하면 단일 소스에 편향될 수 있어, 각 소스에서 최소 1개 청크를 보장한 뒤 나머지를 점수순으로 채웁니다.

### 실패한 시도들
- **Query Decomposition (V7, -5%)**: 하위 질문별 검색이 오히려 노이즈 유발
- **CoRAG (V10, -4%)**: Bounded document pool에서 iterative retrieval은 불필요
- **BM25 Hybrid (V13, -5%)**: 소규모 청크 풀에서 semantic search만으로 충분한 recall 확보

## 기술 스택

- **Embedding**: BAAI/bge-small-en-v1.5 (sentence-transformers, 로컬)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **생성 모델**: OpenAI gpt-4o-mini / gpt-5-mini
- **평가 모델**: OpenAI GPT-5-mini
- **Wikipedia**: REST API + BeautifulSoup (비동기 fetch, JSON 캐싱)
