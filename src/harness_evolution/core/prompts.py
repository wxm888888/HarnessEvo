"""Prompt templates shipped as package data.

Each stage keeps its prompts as Markdown files in a ``prompts/`` directory next
to its code. Templates use ``string.Template`` syntax: ``${name}`` is a
placeholder and ``$$`` is a literal dollar sign. The rendered text is the file
content without its final newline, so a prompt that must end with a newline
keeps an empty last line in the file.

A file may also be split into named sections with ``<!-- section: name -->``
lines when the caller assembles optional parts itself.
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from string import Template

_SECTION_RE = re.compile(r"^<!-- section: ([\w-]+) -->$", re.MULTILINE)


@lru_cache(maxsize=None)
def load_prompt(package: str, name: str) -> str:
    text = resources.files(package).joinpath("prompts", name).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def render_prompt(package: str, template: str, /, **values: object) -> str:
    # Positional-only so templates may use ${name}/${template} placeholders.
    return Template(load_prompt(package, template)).substitute(values)


@lru_cache(maxsize=None)
def load_prompt_sections(package: str, name: str) -> dict[str, str]:
    """Split a sectioned template; each section's text excludes the newline after its marker."""
    text = load_prompt(package, name)
    parts = _SECTION_RE.split(text)
    if parts[0].strip():
        raise ValueError(f"{package}/prompts/{name}: text before the first section marker")
    sections: dict[str, str] = {}
    for key, body in zip(parts[1::2], parts[2::2]):
        if key in sections:
            raise ValueError(f"{package}/prompts/{name}: duplicate section {key}")
        body = body[1:] if body.startswith("\n") else body
        sections[key] = body[:-1] if body.endswith("\n") else body
    return sections


def render_prompt_section(package: str, template: str, section: str, /, **values: object) -> str:
    return Template(load_prompt_sections(package, template)[section]).substitute(values)
