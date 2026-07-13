#!/usr/bin/env bash
# Shared platform tag for release artifact names (linux-x86_64, darwin-arm64, …).

platform_tag() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$arch" in
    x86_64 | amd64) arch="x86_64" ;;
    arm64 | aarch64) arch="aarch64" ;;
  esac
  case "$os" in
    darwin) echo "darwin-${arch}" ;;
    linux) echo "linux-${arch}" ;;
    mingw* | msys* | cygwin* | windows*) echo "windows-${arch}" ;;
    *) echo "${os}-${arch}" ;;
  esac
}
