"""Shared fixtures: a real mock OpenAI server (uvicorn in a thread) on an
ephemeral port, plus config files under tmp_path.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator

import pytest
import uvicorn

from tests.mockserver import MockOpenAIServer, make_app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ServerController:
    def __init__(self, app, port):
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.port = port
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("mock server did not start")
            time.sleep(0.01)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


ServerFactory = Callable[..., _ServerController]


@pytest.fixture
def mock_server_factory() -> Iterator[ServerFactory]:
    """Start mock servers with preset kwargs; every one is stopped at teardown.

    Use it directly for a one-off preset, or via the named fixtures below
    when a test just needs a standard behaviour.
    """
    controllers: list[_ServerController] = []

    def _factory(**preset: object) -> _ServerController:
        ctrl = _ServerController(
            make_app(MockOpenAIServer(first_chunk_ms=20, chunk_ms=20, **preset)),  # type: ignore[arg-type]
            _free_port(),
        )
        controllers.append(ctrl)
        return ctrl

    yield _factory

    for ctrl in reversed(controllers):
        ctrl.stop()


@pytest.fixture
def mock_server(mock_server_factory: ServerFactory) -> _ServerController:
    return mock_server_factory()


@pytest.fixture
def mock_server_failing(mock_server_factory: ServerFactory) -> _ServerController:
    return mock_server_factory(fail_models={"fail-model"})


@pytest.fixture
def mock_server_think_only(mock_server_factory: ServerFactory) -> _ServerController:
    # Thinking model that streams only reasoning_content and no content
    # (the qwen-3.8-27b failure mode on the agentic scenario).
    return mock_server_factory(think_only_models={"qwen-thinker"})


@pytest.fixture
def mock_server_serial_reader(mock_server_factory: ServerFactory) -> _ServerController:
    # Reads one file per tool turn instead of batching parallel calls (the
    # qwen-3.8-27b behaviour): exercised by the multi-turn agentic test.
    return mock_server_factory(serial_reader_models={"qwen-serial"})


@pytest.fixture
def mock_server_empty(mock_server_factory: ServerFactory) -> _ServerController:
    # Empty responder: HTTP 200 stream with zero content AND zero reasoning
    # (the thinking-starved qwen signature: tokens=N, chars=0, no ttft).
    return mock_server_factory(empty_models={"empty-model"})


@pytest.fixture
def mock_server_wrong_answer(mock_server_factory: ServerFactory) -> _ServerController:
    # Answers non-empty but generically ("looks fine") — fails the scenario's
    # fuzzy fact checks (no fib, no 42).
    return mock_server_factory(wrong_answer_models={"wrong-answer"})


@pytest.fixture
def providers_path(tmp_path, mock_server):
    p = tmp_path / "providers.yaml"
    p.write_text(
        f"""providers:
  - name: mock
    base_url: {mock_server.base_url}
    token: test-token
"""
    )
    return p
