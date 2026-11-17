"""llmping — ping-style health probes for OpenAI-compatible LLM servers
(vLLM, SGLang, TGI, llama.cpp, Ollama)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llmping")
except PackageNotFoundError:  # not installed (e.g. a bare source tree)
    __version__ = "0.0.0"
