#!/usr/bin/env bash
# Build proot v5.4.1 with accept/accept4 removed from its seccomp filter.
# Needs: git, gcc, make, pkg-config, libtalloc-dev, uthash-dev (Ubuntu 22.04 tested).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG=v5.4.1
COMMIT=d4b49e2243dee34e41c13dfeca6a3d07e9cf0f00
WORK="${WORK:-$(mktemp -d)}"

git clone -q --depth 1 --branch "$TAG" https://github.com/proot-me/proot.git "$WORK/proot"
cd "$WORK/proot"
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { echo "tag $TAG is not $COMMIT" >&2; exit 1; }
git apply "$HERE/seccomp-unfilter-accept.patch"

# Same two steps as upstream's .github/workflows/release.yml, proot only.
make -s -C src clean loader.elf loader-m32.elf build.h HAS_SWIG= HAS_PYTHON_CONFIG=
LDFLAGS="${LDFLAGS:-} -static" make -s -C src proot HAS_SWIG= HAS_PYTHON_CONFIG=

cp src/proot "$HERE/proot-noaccept"
sha256sum "$HERE/proot-noaccept"
