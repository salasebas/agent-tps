import asyncio
import json
import pytest
from agent_tps_bench.llm_stream_runner import LLMStreamRunner
from agent_tps_bench.models import BenchmarkStatus, TimeoutType


@pytest.mark.asyncio
async def test_llm_stream_success():
    """Tests full streaming parse with chunks, ITL tracking, and usage tokens."""
    sse_body = (
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "!"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}\n\n'
        'data: [DONE]\n\n'
    )

    async def handle_client(reader, writer):
        req = await reader.readuntil(b"\r\n\r\n")
        response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Connection: close\r\n\r\n" + sse_body
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    runner = LLMStreamRunner(base_url=f"http://127.0.0.1:{port}", provider="test")
    chunks_received = []

    res = await runner.run_stream(
        model="mock-gpt",
        prompt="Hi",
        on_chunk=lambda c: chunks_received.append(c),
    )

    server.close()
    await server.wait_closed()

    assert res.status == BenchmarkStatus.SUCCESS
    assert res.timeout_type == TimeoutType.NONE
    assert "".join(chunks_received) == "Hello world!"
    assert res.tokens.input_tokens == 10
    assert res.tokens.output_tokens == 3
    assert res.timings.ttft_ms is not None
    assert res.tps.decode_tps > 0.0


@pytest.mark.asyncio
async def test_llm_stream_rate_limit_429():
    """Tests detection and parsing of HTTP 429 rate limit response."""
    async def handle_client(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        response = (
            "HTTP/1.1 429 Too Many Requests\r\n"
            "Content-Type: application/json\r\n"
            "Retry-After: 3.5\r\n"
            "Connection: close\r\n\r\n"
            '{"error": {"message": "Rate limit reached for requests per minute", "type": "requests"}}'
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    runner = LLMStreamRunner(base_url=f"http://127.0.0.1:{port}", provider="test")
    res = await runner.run_stream(model="mock-gpt", prompt="Hi")

    server.close()
    await server.wait_closed()

    assert res.status == BenchmarkStatus.RATE_LIMITED
    assert res.timeout_type == TimeoutType.RATE_LIMIT_429
    assert res.retry_after_s == 3.5
    assert "Rate limit" in (res.error_message or "")


@pytest.mark.asyncio
async def test_llm_stream_ttft_timeout():
    """Tests TTFT timeout when the server delays sending the first chunk."""
    async def handle_client(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
        )
        await writer.drain()
        # Sleep longer than the TTFT timeout
        await asyncio.sleep(1.0)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    runner = LLMStreamRunner(base_url=f"http://127.0.0.1:{port}", provider="test")
    res = await runner.run_stream(
        model="mock-gpt",
        prompt="Hi",
        ttft_timeout_s=0.2,
    )

    server.close()
    await server.wait_closed()

    assert res.status == BenchmarkStatus.TIMEOUT
    assert res.timeout_type == TimeoutType.TTFT_TIMEOUT
