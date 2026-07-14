# Justfile for Pulsing

# Default target
default: dev

# =============================================================================
# Development
# =============================================================================
# Profiles (root Cargo.toml):
#   - default / `just dev` → [profile.dev]  (debug=1, fast link)
#   - `just build-binary release=release` → [profile.release] (opt-level=s, thin LTO, strip)
# Set DEBUG=1 is a no-op alias for the default fast path (mirrors probing Makefile).

# Install Python package in development mode (wheel path: extension-module)
# Uses [profile.dev] — prefer this for day-to-day iteration (faster than --release).
dev:
    @echo "==> Path A: maturin develop (extension-module, profile.dev)..."
    maturin develop
    @echo "==> Building benchmarks..."
    maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
    @echo "Ready! Use: python -m pulsing.cli  |  just build-release for wheel + binary"

# Same as `dev` but install with [profile.release] (smaller .so, slower compile)
dev-release:
    @echo "==> Path A: maturin develop --release (profile.release)..."
    maturin develop --release
    @echo "==> Building benchmarks (release)..."
    maturin develop --release --manifest-path crates/pulsing-bench-py/Cargo.toml
    @echo "Ready! (release profile)"

# Build pulsing-cli debug binary (RustPython path; profile.dev)
dev-binary:
    just build-binary

# Path A: release wheels for PyPI (extension-module via pyproject.toml)
build-wheel:
    bash scripts/build-wheel.sh --release

# Path B: ``pulsing`` single binary (RustPython VM; no libpython / no PYO3_PYTHON)
# release=release → smaller binary via [profile.release] (strip + thin LTO + opt-level=s)
build-binary release="" package="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    [ "{{release}}" = "release" ] && args+=(--release)
    [ "{{package}}" = "package" ] && args+=(--package)
    bash scripts/build-binary.sh "${args[@]}"

# Path A + B: wheels + single binary (current platform)
build-release package="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--release)
    [ "{{package}}" = "package" ] && args+=(--package)
    bash scripts/build-release.sh "${args[@]}"

# Both distribution artifacts (alias)
build-all package="":
    just build-release package={{package}}

# Build release wheels (alias for wheel-only release)
build: build-wheel

# =============================================================================
# Testing & QA
# =============================================================================

# 提交前本地检查 (格式 + lint + 测试)
check: check-fmt lint test
    @echo ""
    @echo "✅ All checks passed! Ready to commit."

# 快速检查 (仅格式和 lint，不运行测试)
check-quick: check-fmt lint
    @echo ""
    @echo "✅ Format and lint checks passed!"

# Python sources for ruff (docs markdown uses docs/pyproject.toml separately)
ruff_paths := "python tests examples benchmarks crates/pulsing-bench-py"

# 检查代码格式 (不修改)
check-fmt:
    @echo "==> Checking Rust format..."
    cargo fmt --all -- --check
    @echo "==> Checking Python format..."
    ruff format --check {{ruff_paths}}

# Run all tests
test: test-rust test-python

# Run Rust tests (pulsing-py via maturin; pulsing-cli via separate `-p` graph)
test-rust:
    cargo test --workspace --exclude pulsing-bench-py --exclude pulsing-py --exclude pulsing-cli
    cargo test -p pulsing-cli

# Run Python tests
test-python:
    pytest tests/python --ignore=tests/python/test_chaos.py

# Run Chaos tests (separated because they are slower/flakier)
test-chaos:
    pytest tests/python/test_chaos.py

# Run Queue & Topic chaos tests (concurrent, join/leave, mixed workload)
test-queue-topic-chaos:
    pytest tests/python/test_queue_topic_chaos.py -v -s

# Format all code (Rust + Python)
fmt:
    cargo fmt
    ruff format {{ruff_paths}}

# Lint all code
lint:
    cargo clippy --workspace --exclude pulsing-py --exclude pulsing-bench-py --all-targets -- -D warnings
    ruff check {{ruff_paths}}

# =============================================================================
# Coverage (本地查看覆盖率)
# =============================================================================

# Run all coverage reports
cov: cov-rust cov-python
    @echo ""
    @echo "Coverage reports generated!"
    @echo "  Rust:   target/llvm-cov/html/index.html"
    @echo "  Python: htmlcov/index.html"
    @echo ""
    @echo "Run 'just cov-open' to open in browser"

# Rust coverage with HTML report
cov-rust:
    @echo "Running Rust tests with coverage..."
    cargo llvm-cov --workspace --exclude pulsing-py --exclude pulsing-bench-py --html
    @echo "Report: target/llvm-cov/html/index.html"

# Rust coverage summary (terminal only, no HTML)
cov-rust-summary:
    cargo llvm-cov --workspace --exclude pulsing-py --exclude pulsing-bench-py

# Rust coverage with nightly (支持 #[coverage(off)] 标记，但可能不稳定)
cov-rust-nightly:
    @echo "Running Rust tests with coverage (nightly)..."
    cargo +nightly llvm-cov --workspace --exclude pulsing-py --exclude pulsing-bench-py --html
    @echo "Report: target/llvm-cov/html/index.html"

# Python coverage with HTML report
cov-python:
    @echo "Running Python tests with coverage..."
    pytest tests/python --ignore=tests/python/test_chaos.py --cov=python/pulsing --cov-report=html --cov-report=term
    @echo "Report: htmlcov/index.html"

# Open coverage reports in browser (macOS/Linux)
cov-open:
    #!/usr/bin/env bash
    if [ -f target/llvm-cov/html/index.html ]; then \
        echo "Opening Rust coverage report..."; \
        open target/llvm-cov/html/index.html 2>/dev/null || xdg-open target/llvm-cov/html/index.html 2>/dev/null; \
    fi
    if [ -f htmlcov/index.html ]; then \
        echo "Opening Python coverage report..."; \
        open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null; \
    fi

# =============================================================================
# CI 环境准备 (各环境不同，统一使用 uv)
# =============================================================================

# --- 公共工具安装 ---

# 安装 uv (如果不存在)
ensure-uv:
    #!/usr/bin/env bash
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        echo "==> uv already installed"
    else
        echo "==> Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

# 安装 Rust (如果不存在)
ensure-rust:
    #!/usr/bin/env bash
    export PATH="$HOME/.cargo/bin:$PATH"
    if command -v rustc &> /dev/null; then
        echo "==> Rust already installed"
    else
        echo "==> Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi

# --- Manylinux (CentOS) 环境准备 ---
ci-setup-manylinux: ensure-rust ensure-uv
    #!/usr/bin/env bash
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    yum install -y gcc gcc-c++ openssl-devel perl-IPC-Cmd libffi-devel
    uv python install 3.10
    uv tool install maturin
    uv tool install pytest
    echo "==> Setup complete!"

# --- macOS 环境准备 ---
ci-setup-macos: ensure-rust ensure-uv
    #!/usr/bin/env bash
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    uv tool install maturin
    uv tool install pytest
    echo "==> Setup complete!"

# --- Fedora 环境准备 ---
ci-setup-fedora python_version="3.12": ensure-uv
    #!/usr/bin/env bash
    export PATH="$HOME/.local/bin:$PATH"
    # Install build dependencies
    dnf install -y gcc gcc-c++ openssl-devel libffi-devel
    # Use uv to install Python (consistent with manylinux setup)
    uv python install {{python_version}}
    uv tool install pytest
    echo "==> Setup complete!"

# --- Debian/Ubuntu 环境准备 ---
ci-setup-debian: ensure-uv
    #!/usr/bin/env bash
    export PATH="$HOME/.local/bin:$PATH"
    uv tool install pytest
    echo "==> Setup complete!"

# =============================================================================
# CI 构建和测试 (统一命令)
# =============================================================================

# 构建 wheel + 单文件二进制 (CI / 发布)
ci-build manylinux="" package="":
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    args=(--release)
    if [ "{{manylinux}}" = "true" ]; then
        args+=(--manylinux)
    fi
    if [ "{{package}}" = "package" ]; then
        args+=(--package)
    fi
    bash scripts/build-release.sh "${args[@]}"
    echo "==> Build complete: dist/*.whl + dist/bin/pulsing-*"

# 仅构建 wheel（兼容旧调用）
ci-build-wheel manylinux="":
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    args=(--release --binary-only)
    if [ "{{manylinux}}" = "true" ]; then
        args+=(--manylinux)
    fi
    bash scripts/build-release.sh "${args[@]}"

# 测试 wheel (通用)
ci-test:
    #!/usr/bin/env bash
    export PATH="$HOME/.local/bin:$PATH"
    # Install wheel and dependencies using uv (preferred) or pip
    if command -v uv &> /dev/null; then
        uv pip install --system dist/*.whl pytest pytest-asyncio
        # Use same interpreter as above (where wheel was installed); do not use uv run (project venv has no pulsing)
        for py in python3.12 python3.11 python3.10 python3 python; do
            if command -v $py &> /dev/null; then
                $py -m pytest tests/python -v
                exit 0
            fi
        done
        echo "Error: No Python interpreter found"
        exit 1
    else
        # Fallback to pip if uv not available
        pip install dist/*.whl pytest pytest-asyncio
        for py in python3 python3.12 python3.11 python3.10 python; do
            if command -v $py &> /dev/null; then
                $py -m pytest tests/python -v
                exit 0
            fi
        done
        echo "Error: No Python interpreter found"
        exit 1
    fi

# =============================================================================
# 本地模拟 CI 流水线 (Action 命令)
# =============================================================================

# --- macOS ---
action-macos:
    @echo "==> macOS: Setup + Build (wheel + binary) + Test"
    just ci-setup-macos
    just ci-build
    just ci-test

# --- Linux x86-64 ---
action-linux:
    docker run --rm \
        -v {{justfile_directory()}}:/workspace -w /workspace \
        quay.io/pypa/manylinux2014_x86_64 \
        bash -c "curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin && just ci-setup-manylinux && just ci-build manylinux=true package=package"

# --- Linux aarch64 (QEMU) ---
action-linux-aarch64:
    docker run --rm --platform linux/arm64 \
        -v {{justfile_directory()}}:/workspace -w /workspace \
        quay.io/pypa/manylinux2014_aarch64 \
        bash -c "curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin && just ci-setup-manylinux && just ci-build manylinux=true package=package"

# =============================================================================
# Maintenance
# =============================================================================

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    rm -rf **/*.so
    rm -rf **/*.pyd
    rm -rf htmlcov/
    rm -rf .coverage
