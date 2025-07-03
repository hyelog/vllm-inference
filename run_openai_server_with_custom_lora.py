import os
import sys
from typing import Optional

# vLLM의 OpenAI API 서버를 실행하기 위한 argparse 및 메인 함수 임포트
from vllm.entrypoints.openai.api_server import main as run_openai_api_server_main
# from vllm.entrypoints.openai.cli_args import make_arg_parser # main()이 내부적으로 사용

from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
# from vllm.utils import FlexibleArgumentParser # main()이 내부적으로 사용

# --- 1. 커스텀 LoRA 리졸버 정의 ---
class MyCustomLoRAResolver(LoRAResolver):
    def __init__(self, custom_lora_path: str = "/app/lora_adapters"):
        super().__init__()
        self.custom_lora_path = custom_lora_path
        print(f"MyCustomLoRAResolver initialized. Adapters expected in: {self.custom_lora_path}")

    async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
        print(f"Attempting to resolve LoRA '{lora_name}' for base model '{base_model_name}' using MyCustomLoRAResolver.")

        lora_model_path = os.path.join(self.custom_lora_path, lora_name)

        if os.path.exists(lora_model_path) and os.path.isdir(lora_model_path):
            print(f"Found LoRA adapter for '{lora_name}' at '{lora_model_path}'.")
            # LoRARequest의 lora_int_id는 OpenAIServingModels에서 내부적으로 다시 할당합니다.
            return LoRARequest(
                lora_name=lora_name,
                lora_int_id=0, # Will be overwritten by OpenAIServingModels
                lora_local_path=lora_model_path
            )
        else:
            print(f"LoRA adapter for '{lora_name}' not found at '{lora_model_path}' or it is not a directory.")
            return None

def register_custom_lora_resolver():
    custom_resolver_instance = MyCustomLoRAResolver(custom_lora_path="/app/lora_adapters")
    LoRAResolverRegistry.register_resolver("my_custom_resolver", custom_resolver_instance)
    print("MyCustomLoRAResolver has been registered with LoRAResolverRegistry.")
    print(f"Supported LoRA resolvers now: {LoRAResolverRegistry.get_supported_resolvers()}")

def main_with_resolver():
    # 1. 커스텀 LoRA 리졸버 등록
    register_custom_lora_resolver()

    print(f"Original sys.argv: {sys.argv}")
    print(f"Running OpenAI API server with custom LoRA resolver registered.")
    print(f"LoRA adapters are expected at /app/lora_adapters/<lora_name> (e.g., /app/lora_adapters/my_custom_lora_1)")
    print("You can request a LoRA by setting the 'model' field in your API request, e.g., model='my_custom_lora_1'")

    # vllm.entrypoints.openai.api_server.main() 함수를 실행합니다.
    # 이 함수는 내부적으로 sys.argv를 사용하여 인자를 파싱합니다.
    # Dockerfile의 CMD에서 필요한 모든 vLLM 서버 인자들을 전달해야 합니다.
    run_openai_api_server_main()

if __name__ == "__main__":
    main_with_resolver()
