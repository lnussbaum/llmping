"""Providers loading/validation tests.

One config file: providers.yaml. Targets are CLI-only (`--model`,
`--scenario`, `--extra-body`), so they're exercised via the Target model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmping.config import (
    ConfigError,
    Target,
    discover_providers,
    load_providers,
)

PROV = """providers:
  - name: prod
    base_url: https://api.example.fr/v1
    token: s3cr3t
  - name: local
    base_url: http://localhost:8000/v1
"""


@pytest.fixture(autouse=True)
def _clean_llmping_env(monkeypatch):
    """Env fallback must not leak between tests (or from the real shell)."""
    for var in list(os.environ):
        if var.startswith(("LLMPING_BASE_URL", "LLMPING_TOKEN")):
            monkeypatch.delenv(var, raising=False)


def _write(tmp_path: Path, text: str, name: str = "providers.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- providers


def test_providers_valid(tmp_path):
    pc = load_providers(_write(tmp_path, PROV))
    assert [p.name for p in pc.providers] == ["prod", "local"]
    assert pc.providers[0].token == "s3cr3t"
    assert pc.providers[1].token == ""  # default


def test_providers_by_name(tmp_path):
    pc = load_providers(_write(tmp_path, PROV))
    assert pc.by_name("local").base_url == "http://localhost:8000/v1"
    with pytest.raises(ConfigError, match="unknown provider"):
        pc.by_name("nope")


def test_providers_trailing_slash_normalized(tmp_path):
    p = _write(tmp_path, "providers:\n  - name: p\n    base_url: http://x:1/v1/\n")
    assert load_providers(p).providers[0].base_url == "http://x:1/v1"


def test_providers_bad_url(tmp_path):
    p = _write(tmp_path, "providers:\n  - name: p\n    base_url: not-a-url\n")
    with pytest.raises(ConfigError, match="http"):
        load_providers(p)


def test_providers_base_url_from_env(tmp_path, monkeypatch):
    # base_url omitted from YAML → $LLMPING_BASE_URL fills it in
    monkeypatch.setenv("LLMPING_BASE_URL", "http://env.example:9/v1/")
    p = _write(tmp_path, "providers:\n  - name: p\n    token: t\n")
    assert load_providers(p).providers[0].base_url == "http://env.example:9/v1"  # normalized


def test_providers_base_url_per_provider_env_beats_global(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPING_BASE_URL", "http://global.example/v1")
    monkeypatch.setenv("LLMPING_BASE_URL_MY_PROV", "http://per.example/v1")
    p = _write(
        tmp_path,
        "providers:\n  - name: my-prov\n  - name: other\n    base_url: http://yaml.example/v1\n",
    )
    pc = load_providers(p)
    assert pc.providers[0].base_url == "http://per.example/v1"  # per-provider env
    assert pc.providers[1].base_url == "http://yaml.example/v1"  # YAML wins over global env


def test_providers_token_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPING_TOKEN", "global-tok")
    monkeypatch.setenv("LLMPING_TOKEN_PROD", "per-tok")
    p = _write(
        tmp_path,
        "providers:\n  - name: prod\n    base_url: http://x:1/v1\n  - name: local\n    base_url: http://y:1/v1\n",  # noqa: E501
    )
    pc = load_providers(p)
    assert pc.providers[0].token == "per-tok"  # per-provider env wins
    assert pc.providers[1].token == "global-tok"  # global env fallback


def test_providers_yaml_token_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPING_TOKEN_PROD", "env-tok")
    p = _write(
        tmp_path, "providers:\n  - name: prod\n    base_url: http://x:1/v1\n    token: yaml-tok\n"
    )
    assert load_providers(p).providers[0].token == "yaml-tok"


def test_providers_empty_yaml_token_beats_env(tmp_path, monkeypatch):
    # an explicit empty token means "no auth" (Authorization header omitted)
    # — it must NOT be replaced by the env token
    monkeypatch.setenv("LLMPING_TOKEN_PROD", "env-tok")
    p = _write(tmp_path, 'providers:\n  - name: prod\n    base_url: http://x:1/v1\n    token: ""\n')
    assert load_providers(p).providers[0].token == ""


def test_providers_missing_base_url_and_no_env(tmp_path):
    p = _write(tmp_path, "providers:\n  - name: p\n")
    with pytest.raises(ConfigError, match="base_url missing"):
        load_providers(p)


def test_providers_env_base_url_still_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPING_BASE_URL", "not-a-url")
    p = _write(tmp_path, "providers:\n  - name: p\n")
    with pytest.raises(ConfigError, match="http"):
        load_providers(p)


def test_providers_duplicate_name(tmp_path):
    p = _write(
        tmp_path,
        "providers:\n  - name: p\n    base_url: http://x:1/v1\n  - name: p\n    base_url: http://y:1/v1\n",  # noqa: E501
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_providers(p)


def test_providers_unknown_key(tmp_path):
    p = _write(tmp_path, "providers:\n  - name: p\n    base_url: http://x:1/v1\n    apikey: x\n")
    with pytest.raises(ConfigError, match="apikey"):
        load_providers(p)


def test_providers_empty(tmp_path):
    p = _write(tmp_path, "providers: []\n")
    with pytest.raises(ConfigError, match="providers"):
        load_providers(p)


def test_providers_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="providers file not found"):
        load_providers(tmp_path / "nope.yaml")


def test_providers_missing_section(tmp_path):
    p = _write(tmp_path, "something_else: []\n")
    with pytest.raises(ConfigError, match="providers"):
        load_providers(p)


def test_discover_providers_order(tmp_path, monkeypatch):
    explicit = tmp_path / "a.yaml"
    explicit.write_text(PROV)
    env = tmp_path / "b.yaml"
    env.write_text(PROV)
    monkeypatch.setenv("LLMPING_PROVIDERS", str(env))
    assert discover_providers(str(explicit)) == explicit
    assert discover_providers(None) == env
    monkeypatch.delenv("LLMPING_PROVIDERS")
    # no local file → the XDG default (honours $XDG_CONFIG_HOME)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert discover_providers(None) == tmp_path / "xdg" / "llmping" / "providers.yaml"
    # a local ./providers.yaml wins over the XDG default (returned relative
    # to the CWD, so compare resolved paths)
    local = tmp_path / "providers.yaml"
    local.write_text(PROV)
    assert discover_providers(None).resolve() == local.resolve()


def test_discover_providers_xdg_default_home(tmp_path, monkeypatch):
    # Without $XDG_CONFIG_HOME the fallback is ~/.config/llmping/providers.yaml.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    p = discover_providers(None)
    assert p == Path.home() / ".config" / "llmping" / "providers.yaml"


# --------------------------------------------------------------------------- Target (CLI-built)


def test_target_minimal():
    t = Target(model="qwen-3.8-27b")
    assert t.model == "qwen-3.8-27b"
    assert t.scenario == "pingpong"  # the default plain probe
    assert t.extra_body == {}


def test_target_scenario_named():
    # `scenario:` names a code-defined scenario; pingpong is the default.
    assert Target(model="m", scenario="agentic-basic").scenario == "agentic-basic"
    assert Target(model="m", scenario="pingpong").scenario == "pingpong"


def test_target_scenario_empty_rejected():
    with pytest.raises(ConfigError, match="scenario"):
        Target(model="m", scenario="")


def test_target_model_empty_rejected():
    with pytest.raises(ConfigError, match="model"):
        Target(model="")


def test_target_unknown_key_rejected():
    with pytest.raises(ConfigError, match="provider"):
        Target(model="m", provider="p")  # type: ignore[call-arg]


def test_extra_body_must_be_mapping():
    with pytest.raises(ConfigError, match="extra_body"):
        Target(model="m", extra_body=42)  # type: ignore[arg-type]
