import asyncio
import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# --- 1. 커스텀 LoRA 리졸버 정의 ---
class MyCustomLoRAResolver(LoRAResolver):
    """
    사용자 정의 LoRA 리졸버 예시입니다.
    실제 환경에서는 이 클래스 내에 LoRA 어댑터를 로드하고 검증하는 로직을 구현해야 합니다.
    예를 들어, 특정 로컬 경로, S3, HuggingFace Hub 등에서 LoRA 가중치를 가져올 수 있습니다.
    """
    def __init__(self, custom_lora_path: str = "/app/lora_adapters"):
        super().__init__()
        self.custom_lora_path = custom_lora_path
        print(f"MyCustomLoRAResolver initialized. Adapters expected in: {self.custom_lora_path}")
        # 실제로는 여기서 LoRA 어댑터 목록을 미리 스캔하거나 할 수 있습니다.

    async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
        """
        LoRA 어댑터를 이름으로 확인하고 LoRARequest 객체를 반환합니다.
        """
        print(f"Attempting to resolve LoRA '{lora_name}' for base model '{base_model_name}' using MyCustomLoRAResolver.")

        # 예시: lora_name이 "my_custom_lora_1"인 경우 특정 경로의 어댑터를 사용하도록 설정
        # 실제로는 lora_name을 기반으로 어댑터 경로를 동적으로 결정해야 합니다.
        if lora_name == "my_custom_lora_1":
            # LoRA 어댑터 파일들이 있는 실제 경로를 지정해야 합니다.
            # 이 경로는 Docker 이미지 내에 있거나, 볼륨 마운트된 경로일 수 있습니다.
            # adapter_model.bin, adapter_config.json 등이 포함된 디렉토리여야 합니다.
            lora_model_path = os.path.join(self.custom_lora_path, lora_name)

            if os.path.exists(lora_model_path):
                print(f"Found LoRA adapter for '{lora_name}' at '{lora_model_path}'.")
                # LoRARequest 생성 시 lora_local_path에 실제 파일 경로를 제공해야 합니다.
                # lora_id는 vLLM 내부에서 사용될 고유 ID입니다.
                # base_model_path는 현재 vLLM에서 LoRA 어댑터가 어떤 베이스 모델 가중치와 호환되는지
                # 명시적으로 지정하는 필드는 아니지만, 리졸버 구현 시 참고할 수 있습니다.
                return LoRARequest(
                    lora_name=lora_name,
                    lora_int_id=1, # 내부적으로 사용할 정수 ID, LoRA 개수만큼 순차적으로 부여 가능
                    lora_local_path=lora_model_path
                )
            else:
                print(f"LoRA adapter for '{lora_name}' not found at '{lora_model_path}'.")
                return None

        print(f"LoRA adapter for '{lora_name}' not handled by this resolver.")
        return None

# --- 2. 커스텀 LoRA 리졸버 등록 ---
custom_resolver = MyCustomLoRAResolver(custom_lora_path="/app/lora_adapters")
LoRAResolverRegistry.register_resolver("my_custom_resolver", custom_resolver)
print("MyCustomLoRAResolver has been registered with LoRAResolverRegistry.")
print(f"Supported LoRA resolvers: {LoRAResolverRegistry.get_supported_resolvers()}")

# --- 3. vLLM 엔진 및 FastAPI 앱 설정 ---
MODEL_DIR = os.environ.get("MODEL_DIR", "mistralai/Mistral-7B-Instruct-v0.1")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", 256))

engine_args = AsyncEngineArgs(
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    max_num_seqs=MAX_NUM_SEQS,
    enable_lora=True,
    max_loras=5,
    max_lora_rank=64,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    lora_name: Optional[str] = None
    use_beam_search: bool = False
    n: int = 1
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128
    stop: Optional[List[str]] = None

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    request_id = f"cmpl-{random_uuid()}"

    sampling_params = SamplingParams(
        n=request.n,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stop=request.stop,
        use_beam_search=request.use_beam_search,
    )

    final_lora_request: Optional[LoRARequest] = None
    if request.lora_name:
        resolved_lora = await LoRAResolverRegistry.get_resolver("my_custom_resolver").resolve_lora(MODEL_DIR, request.lora_name)
        if resolved_lora:
            final_lora_request = resolved_lora
            print(f"Using resolved LoRA: {final_lora_request.lora_name} with id {final_lora_request.lora_int_id} from {final_lora_request.lora_local_path}")
        else:
            print(f"LoRA '{request.lora_name}' could not be resolved by custom resolver. Using base model.")

    results_generator = engine.generate(
        prompt=request.prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=final_lora_request
    )

    final_output = None
    async for request_output in results_generator:
        if await asyncio.to_thread(request_output.finished):
            final_output = request_output
            break

    if final_output is None:
        return JSONResponse({"error": "Failed to generate text."}, status_code=500)

    generated_texts = [output.text for output in final_output.outputs]
    return JSONResponse({"request_id": request_id, "generated_texts": generated_texts})

@app.on_event("startup")
async def startup_event():
    print("FastAPI application started.")

@app.on_event("shutdown")
async def shutdown_event():
    print("FastAPI application shutting down.")

if __name__ == "__main__":
    print("To run this application, use Uvicorn, e.g.:")
    print("uvicorn main:app --host 0.0.0.0 --port 8000")
    print(f"Default model configured: {MODEL_DIR}")
    print("Make sure your custom LoRA adapters are available at the path specified in MyCustomLoRAResolver (e.g., /app/lora_adapters/my_custom_lora_1).")
