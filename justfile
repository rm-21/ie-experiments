set quiet := true
set shell := ["bash", "-euc"]

[default]
[private]
help:
    @just --list --unsorted

# ── Development ───────────────────────────────────────────────────────────────

[doc('Install dependencies via uv sync')]
[group('dev')]
install:
    @echo '{{ BOLD + GREEN }}install{{ NORMAL }}'
    uv sync --all-packages --all-groups

[doc('Format code with ruff (sort imports + format)')]
[group('dev')]
format: install
    @echo '{{ BOLD + BLUE }}format{{ NORMAL }}'
    uv run ruff check --select I --fix
    uv run ruff format

[doc('Run linters: ty + ruff check')]
[group('dev')]
lint: install
    @echo '{{ BOLD + BLUE }}lint{{ NORMAL }}'
    uv run ty check
    uv run ruff check .

[doc('Run all checks: install, format, lint')]
[group('dev')]
[no-exit-message]
all: install format lint

# ── Clean ─────────────────────────────────────────────────────────────────────

[confirm('Remove all temporary files?')]
[doc('Remove temporary files (__pycache__, .ruff_cache, etc.)')]
[group('clean')]
clean:
    #!/usr/bin/env bash
    echo '{{ BOLD + RED }}clean{{ NORMAL }}'
    find . -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.py[co]' -not -path './.venv/*' -delete 2>/dev/null || true
    find . -type f -name '*~' -delete 2>/dev/null || true
    find . -type f -name '.*~' -delete 2>/dev/null || true
    rm -rf .cache .pytest_cache .ruff_cache htmlcov build dist
    find . -name '*.egg-info' -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
    rm -f .coverage .coverage.*
    echo '{{ BOLD + GREEN }}Cleaning completed successfully.{{ NORMAL }}'

[confirm('Delete every .venv in the workspace?')]
[doc('Remove the workspace .venv and any stray member .venv directories')]
[group('clean')]
clean-venv:
    #!/usr/bin/env bash
    echo '{{ BOLD + RED }}clean-venv{{ NORMAL }}'
    venvs=$(find . -maxdepth 2 -type d -name .venv)
    if [ -n "$venvs" ]; then
        echo "$venvs" | xargs rm -rf
        echo "$venvs" | sed 's/^/  removed /'
        echo '{{ BOLD + GREEN }}Virtual environments deleted.{{ NORMAL }}'
    else
        echo '{{ BOLD + YELLOW }}No virtual environment found.{{ NORMAL }}'
    fi

[confirm('Delete .venv and reinstall from uv.lock?')]
[doc('Rebuild the environment from scratch: clean-venv + install')]
[group('clean')]
reset-venv:
    just --yes clean-venv
    just install
