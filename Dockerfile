# Dockerfile

# 1. CUDA 지원 기본 이미지 선택
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV VLLM_VERSION=0.8.5

# 2. 시스템 패키지 및 Python 설치 (Python 3.9)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev python3-pip git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN python3 -m pip install --no-cache-dir --upgrade pip

# 3. Python 의존성 설치
RUN pip install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# vLLM 및 기타 패키지 설치
RUN pip install --no-cache-dir \
    vllm==${VLLM_VERSION} \
    transformers

# 4. 애플리케이션 코드 복사
WORKDIR /app
COPY ./run_openai_server_with_custom_lora.py /app/run_openai_server_with_custom_lora.py

# (선택 사항) LoRA 어댑터 파일들을 이미지에 포함시키는 경우:
# Docker 빌드 컨텍스트에 lora_adapters 디렉토리가 있다고 가정합니다.
# RUN mkdir -p /app/lora_adapters/my_custom_lora_1
# COPY ./lora_adapters/my_custom_lora_1 /app/lora_adapters/my_custom_lora_1

# 5. 포트 노출 (vLLM OpenAI API 서버 기본 포트)
EXPOSE 8000

# 6. 서버 실행 명령
ENV HF_HOME=/app/huggingface_cache
RUN mkdir -p $HF_HOME

CMD ["python", "/app/run_openai_server_with_custom_lora.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--model", "mistralai/Mistral-7B-Instruct-v0.1", \
     "--tokenizer", "mistralai/Mistral-7B-Instruct-v0.1", \
     "--enable-lora", \
     "--max-loras", "5", \
     "--max-lora-rank", "64" \
    ]
