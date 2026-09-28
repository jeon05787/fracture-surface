# 파손 단면 분석 기반 이미지 분류 시스템

금속 파손 단면 이미지를 입력받아
CNN 모델을 통해 파손 유형을 분류하고,
LLM(Ollama)을 이용하여 결과를 설명해주는 AI 웹 시스템입니다.

---

## 프로젝트 개요

본 프로젝트는 금속 재료의 파손 단면 이미지를 분석하여
파손 원인을 자동으로 예측하고 설명하는 시스템을 구현하는 것을 목표로 합니다.

사용자는 이미지를 업로드하면 다음과 같은 결과를 확인할 수 있습니다.

* 파손 유형 예측
* 각 유형별 유사도
* 신뢰도 (confidence)
* AI 기반 설명 (LLM)

---

## 분류 유형

본 시스템은 다음 4가지 파손 유형을 분류합니다.

* 취성 파괴 (Cleavage)
* 연성 파괴 (Ductile)
* 피로 파괴 (Fatigue)
* 입계 파괴 (Intergranular)

---

## 사용 기술

### Backend

* Python
* FastAPI
* PyTorch
* Ollama (LLM)

### Frontend

* React
* JavaScript
* Tailwind CSS

---

## 프로젝트 구조

```bash
fracture_surface/
├── backend/
│   ├── main.py
│   ├── fractography_cnn_best1.pth
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
└── model/
    ├── train.py
    ├── predict.py
    └── Fractography/
```

---

## 실행 방법 (환경 구축 및 실행 가이드)

예상 소요 시간: 약 30 ~ 40분

---

## 전체 흐름

1. 필수 프로그램 설치
2. 프로젝트 다운로드
3. Backend 환경 구축
4. Frontend 환경 구축
5. Ollama 설치
6. 전체 실행

---

## 1. 필수 프로그램 설치

### 1-1. Anaconda 설치

* https://www.anaconda.com/download 접속
* Windows 64-bit 다운로드
* 설치 시 다음 옵션 체크

  * Just Me 선택
  * 기본 경로 유지
  * **Add Anaconda3 to PATH 체크 (중요)**

설치 후 PC 재시작 권장

---

### 1-2. Node.js 설치

* https://nodejs.org 접속
* LTS 버전 설치

설치 확인:

```bash
node -v
npm -v
```

---

### 1-3. Git 설치 (선택)

* https://git-scm.com/download/win
* 기본 설정으로 설치

---

## 2. 프로젝트 다운로드

```bash
git clone https://github.com/(GitHub계정)/(레포지토리명).git
```

또는 ZIP 다운로드 후 압축 해제

---

## 3. Backend 환경 구축

```bash
cd backend
```

### 가상환경 생성

```bash
conda create -n fanalysis python=3.10 -y
```

※ fanalysis는 프로젝트 전용 가상환경 이름이며, 원하는 이름으로 변경 가능합니다.

---

### 가상환경 활성화

```bash
conda activate fanalysis
```

※ `(fanalysis)` 표시가 나타나면 정상적으로 활성화된 것입니다.
※ 이후 backend 관련 명령어는 해당 가상환경에서 실행해야 합니다.

---

### PyTorch 설치

#### CPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### GPU (CUDA 11.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### 기타 패키지 설치

```bash
pip install -r requirements.txt
```

---

### 모델 파일

학습된 모델 파일(.pth)을 backend 폴더에 위치시켜야 합니다.

```bash
backend/
├── main.py
├── requirements.txt
└── model.pth
```

파일명은 자유롭게 설정할 수 있으며,
main.py의 MODEL_PATH와 동일하게 맞춰야 합니다.

---

### 설치 확인

```bash
python -c "import torch, fastapi, ollama; print('OK')"
```

---

## 4. Frontend 환경 구축

```bash
cd frontend
```

### 패키지 설치

```bash
npm install
```

---

### Tailwind CSS 초기화

```bash
npx tailwindcss init -p
```

※ 이미 `tailwind.config.js` 파일이 존재하는 경우 생략 가능합니다.

---

### 실행 확인

```bash
npm list react
```

---

## 5. Ollama 설치

### 설치

* https://ollama.com 다운로드 후 설치

---

### 모델 다운로드

```bash
ollama pull llama3
```

---

### 확인

```bash
ollama list
```

---

## 6. 전체 실행

👉 총 3개의 터미널 필요

---

### 터미널 1 — Ollama 실행

```bash
ollama serve
```

※ Ollama 서버는 백그라운드에서 계속 실행 상태를 유지해야 합니다.

---

### 터미널 2 — Backend 실행

```bash
cd backend
conda activate fanalysis
uvicorn main:app --reload
```

접속:

```
http://127.0.0.1:8000
```

---

### 터미널 3 — Frontend 실행

```bash
cd frontend
npm start
```

접속:

```
http://localhost:3000
```

---

## 자주 발생하는 오류

| 오류                        | 원인              | 해결 방법                           |
| ------------------------- | --------------- | ------------------------------- |
| conda: command not found  | PATH 미설정        | Anaconda 재설치                    |
| npm: command not found    | Node.js 미설치     | Node.js 설치                      |
| ModuleNotFoundError       | 패키지 미설치         | pip install -r requirements.txt |
| CORS error                | Backend 미실행     | uvicorn 실행 확인                   |
| Cannot find module        | npm install 미실행 | frontend에서 npm install          |
| .pth 파일 없음                | 모델 경로 오류        | backend에 파일 위치 확인               |
| ollama connection refused | Ollama 미실행      | ollama serve 실행                 |

---

## 빠른 실행 요약

```bash
# 1. Ollama
ollama serve

# 2. Backend
cd backend
conda activate fanalysis
uvicorn main:app --reload

# 3. Frontend
cd frontend
npm start
```

브라우저 접속:

```
http://localhost:3000
```
