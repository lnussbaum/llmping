"""Providers loading + the CLI-side target model.

One config file: `providers.yaml` (gitignored) — named endpoints + auth.
There is **no** `config.yaml`: what to ping is given on the command line
(`--provider NAME --model MODEL [--scenario NAME]`), so there are no named
targets to maintain.

    providers := providers!
    provider  := { name!, base_url?, token? }
    target    := { model!, scenario?, extra_body? }   # built by the CLI, not YAML

`base_url` / `token` may be omitted from YAML; they then come from the
environment (per-provider `LLMPING_BASE_URL_<NAME>` / `LLMPING_TOKEN_<NAME>`
override the global `LLMPING_BASE_URL` / `LLMPING_TOKEN`). An explicit YAML
value always wins over the environment. There are **no CLI flags** for
`base_url`/`token` — secrets never appear on a command line (they would leak
into shell history and `ps`).

Unknown keys are rejected at every level except inside ``extra_body``
(typo protection). All failures raise :class:`ConfigError` → CLI exit 2.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from llmping.scenario import DEFAULT_SCENARIO


class ConfigError(Exception):
    """Any config/usage error (CLI maps this to exit code 2)."""


_ENV_NAME_RE = re.compile(r"[^A-Z0-9]+")


def _env_base_url(name: str) -> str | None:
    """`LLMPING_BASE_URL_<NAME>` (name uppercased, non-alnum → `_`), else
    the global `LLMPING_BASE_URL`. None when unset."""
    per = os.environ.get(f"LLMPING_BASE_URL_{_ENV_NAME_RE.sub('_', name.upper())}")
    if per:
        return per
    return os.environ.get("LLMPING_BASE_URL") or None


def _env_token(name: str) -> str | None:
    """`LLMPING_TOKEN_<NAME>`, else the global `LLMPING_TOKEN`. None when
    unset (an explicitly empty token stays empty — no Authorization header)."""
    per = os.environ.get(f"LLMPING_TOKEN_{_ENV_NAME_RE.sub('_', name.upper())}")
    if per is not None:
        return per
    return os.environ.get("LLMPING_TOKEN") or None


class ProviderEntry(BaseModel):
    """One named endpoint + auth, from providers.yaml, with env fallback for
    `base_url` / `token` when the YAML key is missing (or empty).

    The fallback runs **before** field validation, so the fields are always
    plain `str` after construction — callers never see `str | None`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    base_url: str = ""
    token: str = ""

    @model_validator(mode="before")
    @classmethod
    def _env_fallback(cls, data: Any) -> Any:
        # YAML wins over the environment; env fills in what YAML left out.
        # Runs before field validation so the fields are always plain `str`.
        if not isinstance(data, dict):
            return data
        name = data.get("name")
        if not isinstance(name, str) or not name:
            return data  # let name validation report the real problem
        patched = dict(data)
        if not patched.get("base_url"):
            env_url = _env_base_url(name)
            if not env_url:
                env_suffix = _ENV_NAME_RE.sub("_", name.upper())
                raise ConfigError(
                    f"provider {name!r}: base_url missing — set it in providers.yaml "
                    f"or via $LLMPING_BASE_URL_{env_suffix} (or $LLMPING_BASE_URL)"
                )
            patched["base_url"] = env_url
        if patched.get("token") is None:  # missing/null; explicit "" is kept
            env_token = _env_token(name)
            if env_token is not None:
                patched["token"] = env_token
        return patched

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ConfigError(f"provider base_url must be an http(s) URL, got {v!r}")
        return v


class ProvidersConfig(BaseModel):
    """The `providers:` file: named endpoints + auth (gitignored)."""

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> ProvidersConfig:
        seen: set[str] = set()
        for p in self.providers:
            if p.name in seen:
                raise ConfigError(f"duplicate provider name: {p.name!r}")
            seen.add(p.name)
        return self

    def by_name(self, name: str) -> ProviderEntry:
        for p in self.providers:
            if p.name == name:
                return p
        names = ", ".join(p.name for p in self.providers)
        raise ConfigError(f"unknown provider {name!r} — providers: {names}")


class Target(BaseModel):
    """One thing to ping: a bare API model name, the scenario to run, an
    optional extra request body, and optional inert `padding` to grow the
    prompt — resolved against the run's single provider. Built by the CLI
    from `--model`/`--scenario`/`--extra-body`/`--pad` — it never comes
    from YAML. `scenario` defaults to `pingpong`, the plain probe.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    # Registry name of a code-defined scenario — defaults to `pingpong`, the
    # plain probe (one turn, no tools). Every target runs a scenario.
    scenario: str = DEFAULT_SCENARIO
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Inert characters appended to the first user message, to artificially
    # grow the prompt (e.g. measure token accounting at a given size). 0 = off.
    padding: int = Field(default=0, ge=0)

    @model_validator(mode="wrap")
    @classmethod
    def _as_config_error(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Target:
        # Target is built by the CLI from flag values; every validation
        # failure (unknown key, wrong type, empty model) must be a
        # ConfigError so the CLI maps it to exit 2 — pydantic's native
        # ValidationError would escape as a traceback instead.
        try:
            result = handler(value)
        except ValidationError as e:
            raise ConfigError(f"invalid target: {e}") from e
        return cast(Target, result)

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        if not v.strip():
            raise ConfigError("target model must be a non-empty name")
        return v

    @field_validator("scenario")
    @classmethod
    def _scenario(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ConfigError("target scenario must be a non-empty name")
        return v

    @field_validator("extra_body")
    @classmethod
    def _extra(cls, v: Any) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ConfigError("extra_body must be a JSON object")
        return v


def discover_providers(explicit: str | None) -> Path:
    """Discovery order: --providers PATH → $LLMPING_PROVIDERS →
    ./providers.yaml → ~/.config/llmping/providers.yaml (XDG config dir
    honoured: $XDG_CONFIG_HOME/llmping/providers.yaml)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("LLMPING_PROVIDERS")
    if env:
        return Path(env)
    local = Path("providers.yaml")
    if local.is_file():
        return local
    config_dir = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(config_dir).expanduser() / "llmping" / "providers.yaml"


def load_providers(path: Path | str) -> ProvidersConfig:
    """Load and validate the providers file. Raises ConfigError on any problem."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"providers file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict) or "providers" not in raw:
        raise ConfigError(f"{path}: missing required 'providers:' list")
    if not raw.get("providers"):
        raise ConfigError(f"{path}: 'providers:' must be non-empty")
    try:
        return ProvidersConfig.model_validate(raw)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"{path}: invalid providers: {e}") from e
