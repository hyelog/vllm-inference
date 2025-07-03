# Dockerfile

# 1. CUDA 지원 기본 이미지 선택
# vLLM v0.8.5는 PyTorch 2.0.1 ~ 2.1.x 와 호환될 가능성이 높습니다.
# CUDA 11.8 또는 12.1 이미지를 사용하는 것이 일반적입니다.
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
# PyTorch 2.1.2 (CUDA 12.1) - vLLM v0.8.5와 호환 가능성 있음. 정확한 버전은 vLLM 문서를 따르세요.
RUN pip install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# vLLM 및 기타 패키지 설치
RUN pip install --no-cache-dir \
    vllm==${VLLM_VERSION} \
    fastapi \
    uvicorn[standard] \
    pydantic \
    transformers

# 4. 애플리케이션 코드 복사
WORKDIR /app
COPY ./main.py /app/main.py

# (선택 사항) LoRA 어댑터 파일들을 이미지에 포함시키는 경우:
# Docker 빌드 컨텍스트에 lora_adapters 디렉토리가 있다고 가정하고,
# 그 안에 my_custom_lora_1 이라는 어댑터가 있다고 가정합니다.
# RUN mkdir -p /app/lora_adapters/my_custom_lora_1
# COPY ./lora_adapters/my_custom_lora_1 /app/lora_adapters/my_custom_lora_1
# 실제 어댑터 경로와 이름에 맞게 수정해야 합니다.
# 또는, 실행 시점에 볼륨 마운트를 통해 제공할 수 있습니다.

# 5. 포트 노출
EXPOSE 8000

# 6. 서버 실행 명령
ENV MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.1"
# ENV MODEL_DIR="EleutherAI/gpt-neo-125m" # 작은 모델로 테스트 시

ENV HF_HOME=/app/huggingface_cache
RUN mkdir -p $HF_HOME

# NCCL 소켓 인터페이스 설정 (다중 GPU 환경에서 필요할 수 있음)
# ENV NCCL_SOCKET_IFNAME=eth0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
