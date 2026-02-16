.PHONY: help serve build clean install dev-install check-links sync

help:
	@echo "Available commands:"
	@echo "  make serve        - Serve documentation locally (auto-installs deps, livereload on)"
	@echo "  make build        - Build documentation (auto-installs deps)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make check-links  - Check for broken links"
	@echo "  make sync         - Sync dependencies with pyproject.toml"
	@echo ""
	@echo "  If livereload does not refresh on file change, try:"
	@echo "    WATCHDOG_USE_POLLING=1 make serve"

sync:
	uv sync --all-extras

serve:
	@echo "Installing Pulsing in development mode..."
	cd .. && uv run maturin develop
	uv run mkdocs serve --livereload -w src -w mkdocs.yml -w overrides

build:
	@echo "Installing Pulsing in development mode..."
	cd .. && uv run maturin develop
	uv run mkdocs build

clean:
	rm -rf site/
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

check-links:
	uv run mkdocs-linkcheck src
