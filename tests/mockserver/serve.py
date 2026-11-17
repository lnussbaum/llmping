"""Run the mock OpenAI server with uvicorn (CI loopback target).

    python -m tests.mockserver.serve --port 8321

Uses `LLMPING_MOCK_FIRST_CHUNK_MS` / `LLMPING_MOCK_CHUNK_MS` to tune the
ground-truth chunk timing, and `LLMPING_MOCK_FAIL` (comma-separated model
names) to inject 503s. Model-behaviour presets (comma-separated names):
`LLMPING_MOCK_SERIAL` — one file read per tool turn (qwen);
`LLMPING_MOCK_THINK_ONLY` — streams reasoning with no visible answer;
`LLMPING_MOCK_EMPTY` — HTTP 200 with empty content *and* reasoning
(the thinking-starved, chars=0 signature); `LLMPING_MOCK_WRONG_ANSWER` —
a plausible-sounding reply that fails the scenario's fuzzy fact checks.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from tests.mockserver import MockOpenAIServer, make_app


def _names(env: str) -> set[str]:
    return {m for m in os.environ.get(env, "").split(",") if m}


def main() -> None:
    ap = argparse.ArgumentParser(description="llmping deterministic mock OpenAI server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LLMPING_MOCK_PORT", "8321")))
    ap.add_argument("--host", default=os.environ.get("LLMPING_MOCK_HOST", "127.0.0.1"))
    args = ap.parse_args()

    server = MockOpenAIServer(
        first_chunk_ms=float(os.environ.get("LLMPING_MOCK_FIRST_CHUNK_MS", "25")),
        chunk_ms=float(os.environ.get("LLMPING_MOCK_CHUNK_MS", "25")),
        fail_models=_names("LLMPING_MOCK_FAIL"),
        think_only_models=_names("LLMPING_MOCK_THINK_ONLY"),
        serial_reader_models=_names("LLMPING_MOCK_SERIAL"),
        empty_models=_names("LLMPING_MOCK_EMPTY"),
        wrong_answer_models=_names("LLMPING_MOCK_WRONG_ANSWER"),
    )
    uvicorn.run(make_app(server), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
