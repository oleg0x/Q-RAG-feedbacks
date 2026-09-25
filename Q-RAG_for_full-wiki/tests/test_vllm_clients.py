import asyncio

from vLLM_clients import AsyncVllmClient, BasicVllmClient, SyncVllmClient


def test_sync_client_maps_thinking_to_chat_template_kwargs():
    client = SyncVllmClient(
        llm="model",
        base_url="http://localhost:8000",
        system_prompt="system",
        thinking=False,
    )
    try:
        assert client.payload["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        assert "thinking" not in client.payload
    finally:
        client.close()


def test_async_client_maps_thinking_to_chat_template_kwargs():
    client = AsyncVllmClient(
        model="model",
        base_url="http://localhost:8000",
        system_prompt="system",
        thinking=False,
    )
    assert client.payload["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "thinking" not in client.payload


def test_basic_client_ignores_proxy_environment_by_default(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")
    client = BasicVllmClient(
        model="model",
        base_url="http://127.0.0.1:8000",
        max_tokens=8,
        thinking=False,
        sys_prompt="system",
    )
    try:
        assert client._sync_http_client._trust_env is False
        assert client._async_http_client._trust_env is False
    finally:
        client._client.close()
        asyncio.run(client._async_client.close())
