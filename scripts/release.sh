#!/usr/bin/env bash
# Build the family-standard release assets into dist/.
# Usage: scripts/release.sh v0.1.0
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:?usage: scripts/release.sh vX.Y.Z}"
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "invalid version: $VERSION" >&2; exit 1; }

for tool in git go zip tar sha256sum sed find touch awk; do
  command -v "$tool" >/dev/null || { echo "missing release tool: $tool" >&2; exit 1; }
done

NAME=Dirtybird-Go-GPU-Miner
BIN=go-gpu-miner
COMMIT="${SOURCE_COMMIT:-$(git rev-parse --short=12 HEAD)}"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}"
LDFLAGS="-s -w -X main.commit=${COMMIT}"
RELEASE_FILES=(README.md DEPLOY.md LICENSE THIRD-PARTY-LICENSES internal/astrobwt/LICENSE-DERO.txt internal/astrobwt/LICENSE-GO.txt internal/gpu/spirv/LICENSE-Dirtybird-CUDA-Miner.txt)

for file in config.example.json start.bat script.sh "${RELEASE_FILES[@]}"; do
  [ -f "$file" ] || { echo "missing release input: $file" >&2; exit 1; }
done

rm -rf dist
mkdir -p dist

build() {
  local goos="$1" goarch="$2" goamd64="$3" output="$4"
  if [ -n "$goamd64" ]; then
    CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" GOAMD64="$goamd64" \
      go build -buildvcs=false -trimpath -ldflags "$LDFLAGS" -o "$output" ./cmd/go-gpu-miner
  else
    CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" \
      go build -buildvcs=false -trimpath -ldflags "$LDFLAGS" -o "$output" ./cmd/go-gpu-miner
  fi
}

normalize_times() {
  find "$1" -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +
}

archive_tar() {
  local output="$1" root="$2"
  tar --sort=name --mtime="@${SOURCE_DATE_EPOCH}" \
    --owner=0 --group=0 --numeric-owner \
    -czf "$output" -C dist "$root"
}

while read -r platform goos goarch goamd64 suffix launcher; do
  [ "$goamd64" = - ] && goamd64=""
  [ "$suffix" = - ] && suffix=""
  stage="dist/${NAME}-${platform}-${VERSION}"
  mkdir -p "$stage"
  echo "== ${platform} (${goos}/${goarch}) =="
  build "$goos" "$goarch" "$goamd64" "${stage}/${BIN}${suffix}"
  cp config.example.json "${stage}/config.json"
  cp "${RELEASE_FILES[@]}" "$stage/"
  cp "$launcher" "$stage/"
  normalize_times "$stage"

  if [ "$goos" = windows ]; then
    (cd dist && zip -Xqr "${NAME}-${platform}-${VERSION}.zip" "${NAME}-${platform}-${VERSION}")
  else
    chmod 0755 "${stage}/${BIN}" "${stage}/${launcher}"
    archive_tar "dist/${NAME}-${platform}-${VERSION}.tar.gz" "${NAME}-${platform}-${VERSION}"
  fi
  rm -rf "$stage"
done <<'MATRIX'
win64 windows amd64 v3 .exe start.bat
amd64 linux amd64 v3 - script.sh
arm64 linux arm64 - - script.sh
macos-arm64 darwin arm64 - - script.sh
MATRIX

HIVE_NAME=dirtybird-go-gpu-miner
HIVE_ASSET="${HIVE_NAME}-${VERSION}.hiveos_mmpos.amd64.tar.gz"
HIVE_STAGE="dist/${HIVE_NAME}"
for file in config/h-manifest.conf config/h-config.sh config/h-run.sh config/h-stats.sh config/mmp-stats.sh; do
  [ -f "$file" ] || { echo "missing HiveOS input: $file" >&2; exit 1; }
done
mkdir -p "$HIVE_STAGE"
build linux amd64 v3 "${HIVE_STAGE}/${BIN}"
cp config/h-manifest.conf config/h-config.sh config/h-run.sh config/h-stats.sh config/mmp-stats.sh "$HIVE_STAGE/"
cp LICENSE THIRD-PARTY-LICENSES internal/astrobwt/LICENSE-DERO.txt internal/astrobwt/LICENSE-GO.txt internal/gpu/spirv/LICENSE-Dirtybird-CUDA-Miner.txt "$HIVE_STAGE/"
sed -i "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=${VERSION#v}/" "${HIVE_STAGE}/h-manifest.conf"
chmod 0755 "${HIVE_STAGE}/${BIN}" "${HIVE_STAGE}"/h-*.sh "${HIVE_STAGE}/mmp-stats.sh"
normalize_times "$HIVE_STAGE"
archive_tar "dist/${HIVE_ASSET}" "$HIVE_NAME"
rm -rf "$HIVE_STAGE"

for archive in dist/*.tar.gz; do
  tar --numeric-owner -tvzf "$archive" | awk -v archive="$archive" '
    $2 != "0/0" {
      print "non-normalized owner in " archive ": " $0 > "/dev/stderr"
      bad = 1
    }
    END { exit bad }
  '
done

(cd dist && sha256sum ./*.zip ./*.tar.gz | sed 's| [ *]\./|  |' > SHA256SUMS.txt)
echo "release assets:"
cat dist/SHA256SUMS.txt
