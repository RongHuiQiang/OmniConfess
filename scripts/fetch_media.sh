#!/usr/bin/env bash
# Download the OmniHalluBench media files (images / videos / audio) and unpack
# them into place, so that every relative path in the annotation JSONs resolves
# with no --data_root flag.
#
#   bash scripts/fetch_media.sh              # all four multimodal subsets, 3.24 GB
#   bash scripts/fetch_media.sh cmm phd      # only the subsets you need
#
# Reproducing the main table does NOT need any of this: the aggregator recomputes
# every number from the generations already committed under results_shipped/.
# Media are only required to re-run generation, which needs model weights anyway.
#
# While the repository is private, anonymous download will not work. Either
# authenticate the GitHub CLI once (`gh auth login`) or export a token with repo
# read scope (`export GITHUB_TOKEN=ghp_...`); the script uses whichever it finds.
set -euo pipefail

REPO="RongHuiQiang/OmniConfess"
TAG="media-v1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}"
CACHE="${REPO_ROOT}/.media_cache"

ALL_SUBSETS=(cmm avhbench haloquest phd)
SUBSETS=("$@")
if [ ${#SUBSETS[@]} -eq 0 ]; then SUBSETS=("${ALL_SUBSETS[@]}"); fi

# sha256 of each tarball, so a truncated download fails loudly instead of
# unpacking a partial subset.
sha_for() {
  case "$1" in
    cmm)       echo ffd16a73dadb2283f091e440dbc9bde458a8ebceca47fcd1b1f8495708e65e8d ;;
    avhbench)  echo fcd207628cf554fef489cc36b24f3381d3969977bbcf11686c28ef5f0a95c874 ;;
    haloquest) echo 05e7470040943d0d53ebebefd2f08f282fc38eaa81a9baea0247786e1ad211cc ;;
    phd)       echo 774265be7aa6d12142ae03d9cb271d18dbfe3a7a8242a78544dfdc72845b148e ;;
    *) return 1 ;;
  esac
}

# expected media file count per subset, checked after extraction
count_for() {
  case "$1" in
    cmm)       echo 1320 ;;
    avhbench)  echo 645 ;;
    haloquest) echo 246 ;;
    phd)       echo 625 ;;
    *) return 1 ;;
  esac
}

download() {  # $1 = asset name, $2 = output path
  local name="$1" out="$2"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" -R "$REPO" -p "$name" -O "$out" --clobber
  else
    local url="https://github.com/${REPO}/releases/download/${TAG}/${name}"
    local auth=()
    if [ -n "${GITHUB_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}"); fi
    curl -fL --retry 3 --retry-delay 5 -C - "${auth[@]}" -o "$out" "$url"
  fi
}

mkdir -p "$CACHE"
for ds in "${SUBSETS[@]}"; do
  want_sha="$(sha_for "$ds")" || {
    echo "unknown subset: $ds (choose from ${ALL_SUBSETS[*]})" >&2; exit 2; }
  name="omnihallubench_media_${ds}.tar"
  tarball="${CACHE}/${name}"

  if [ -f "$tarball" ] && [ "$(sha256sum "$tarball" | cut -d' ' -f1)" = "$want_sha" ]; then
    echo "[$ds] cached tarball already verified"
  else
    echo "[$ds] downloading ${name} ..."
    download "$name" "$tarball"
    got="$(sha256sum "$tarball" | cut -d' ' -f1)"
    if [ "$got" != "$want_sha" ]; then
      echo "[$ds] CHECKSUM MISMATCH" >&2
      echo "  expected $want_sha" >&2
      echo "  got      $got" >&2
      echo "  delete ${tarball} and retry" >&2
      exit 1
    fi
  fi

  echo "[$ds] unpacking into ${DEST} ..."
  tar -xf "$tarball" -C "$DEST"

  got_n="$(find "${DEST}/OmniHalluBench/${ds}" -type f ! -name '*.json' | wc -l)"
  want_n="$(count_for "$ds")"
  if [ "$got_n" -ne "$want_n" ]; then
    echo "[$ds] WARNING: expected ${want_n} media files on disk, found ${got_n}" >&2
  else
    echo "[$ds] ok, ${got_n} media files in place"
  fi
done

cat <<'EOF'

Done. The tarballs are kept under .media_cache/ (gitignored) so a re-run does not
re-download; delete that directory to reclaim the space.

Two HaloQuest images are absent upstream and are therefore not in the tarball
(3_2718_4502888288_e4860b29c3_o.jpg, 4_3256_2789033013_c2f79a2bec_o.jpg). The
3 affected rows out of 365 are documented in KNOWN_ISSUES.md.
EOF
