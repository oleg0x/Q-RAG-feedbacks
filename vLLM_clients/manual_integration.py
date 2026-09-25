"""Explicit manual connectivity check for a live vLLM server.

This module is intentionally not collected by pytest and never runs on import.
"""

import argparse
import asyncio
import os

from vLLM_clients import AsyncVllmClient, SyncVllmClient, extract_response_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--async-client", action="store_true")
    parser.add_argument("--prompt", default="Reply with exactly: OK")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or VLLM_BASE_URL is required")

    common = {
        "base_url": args.base_url,
        "system_prompt": "Follow the user's output-format instruction exactly.",
        "api_key": args.api_key,
        "thinking": False,
        "max_tokens": 8,
        "temperature": 0.0,
    }
    if args.async_client:
        client = AsyncVllmClient(model=args.model, **common)
        response = asyncio.run(client.generate_batch([args.prompt]))[0]
    else:
        with SyncVllmClient(llm=args.model, **common) as client:
            response = client.chat_completion(args.prompt)
    print(extract_response_text(response))


if __name__ == "__main__":
    main()
