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

# Pillow-simd 설치 (이미지 처리 성능 향상)
RUN pip uninstall -y Pillow && \
    CC="cc -mavx2" pip install -U --force-reinstall Pillow-simd --no-cache-dir

# vLLM 및 기타 패키지 설치
RUN pip install --no-cache-dir \
    vllm==${VLLM_VERSION} \
    transformers

# 4. 애플리케이션 코드 복사
WORKDIR /app
COPY ./run_multimodal_server.py /app/run_multimodal_server.py

# 5. 포트 노출 (vLLM OpenAI API 서버 기본 포트)
EXPOSE 8000

# 6. 서버 실행 명령
ENV HF_HOME=/app/huggingface_cache
RUN mkdir -p $HF_HOME

CMD ["python", "/app/run_multimodal_server.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--model", "llava-hf/llava-1.5-7b-hf", \
     "--tokenizer", "llava-hf/llava-1.5-7b-hf", \
     # "--image-input-type", "pixel_values", # vLLM이 자동 감지 시도
     # "--chat-template", "/path/to/llava_chat_template.jinja", # 필요시 LLaVA 챗 템플릿 경로 지정
     "--disable-log-stats" \
    ]
