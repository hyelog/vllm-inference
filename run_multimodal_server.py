import sys
from vllm.entrypoints.openai.api_server import main as run_openai_api_server_main

def main():
    print("Starting vLLM OpenAI API server for multimodal (LLaVA) and text-only requests.")
    print("Ensure the model specified in the CMD (or via CLI override) is a LLaVA model.")
    print("Example LLaVA model: 'llava-hf/llava-1.5-7b-hf'")
    print("Multimodal requests should use the OpenAI content array format for messages.")
    print("Text-only requests can be sent as standard text prompts.")

    # vLLM의 OpenAI API 서버 메인 함수를 직접 호출합니다.
    # 모든 CLI 인자는 Dockerfile의 CMD나 docker run 명령어에서 전달됩니다.
    run_openai_api_server_main()

if __name__ == "__main__":
    main()
