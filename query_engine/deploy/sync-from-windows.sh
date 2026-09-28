#!/usr/bin/env bash
# sync-from-windows.sh — copy src/query_engine/ from this Windows machine (Git Bash)
# to orin11 staging dir ~/qe-stage/query_engine/ (NOT the git root, NOT the
# services dir). Uses tar | ssh because Git Bash has no rsync.
#
# Usage (Git Bash):  bash /c/Users/flash/hd-work/query-engine/src/query_engine/deploy/sync-from-windows.sh
# Then ask the user to run on orin11 (after reading install.sh):
#   sudo bash ~/qe-stage/query_engine/deploy/install.sh ~/qe-stage/query_engine
set -euo pipefail

readonly REMOTE=cluster@192.168.1.239
readonly REMOTE_DIR=/home/cluster/qe-stage/query_engine

die() { printf '\nABORT: %s\n' "$*" >&2; exit 1; }

SRC="$(cd -- "$(dirname -- "$0")/.." && pwd)"
[[ -f "$SRC/nicu_qe/__init__.py" ]] || die "$SRC is not the query_engine source dir"
[[ -f "$SRC/deploy/install.sh" ]] || die "$SRC/deploy/install.sh missing"

# Shell scripts and unit files must have LF endings on Linux.
crlf="$(grep -l $'\r' "$SRC"/deploy/*.sh "$SRC"/deploy/*.service "$SRC"/deploy/sudoers-* "$SRC"/deploy/env.qe.example 2>/dev/null || true)"
[[ -z "$crlf" ]] || die "CRLF line endings in: $crlf (convert with: sed -i 's/\r\$//' <file>)"

echo "==> source : $SRC"
echo "==> target : $REMOTE:$REMOTE_DIR"

# Remote side: extract into a fresh .new dir, then swap. The only rm -rf targets
# are asserted to be exactly $REMOTE_DIR and $REMOTE_DIR.new.
remote_script='set -eu
d=/home/cluster/qe-stage/query_engine
case "$d" in /home/cluster/qe-stage/query_engine) ;; *) echo "bad dir $d" >&2; exit 1;; esac
mkdir -p /home/cluster/qe-stage
rm -rf -- "$d.new"
mkdir -p -- "$d.new"
tar -xf - -C "$d.new"
rm -rf -- "$d"
mv -- "$d.new" "$d"
echo "remote: extracted $(find "$d" -type f | wc -l) files into $d"'

tar -C "$SRC" -cf - \
  --exclude='.venv*' --exclude='*.egg-info' --exclude=.pytest_cache \
  --exclude=__pycache__ --exclude=.env.qe --exclude='*.pid' --exclude=.git \
  . | ssh -o BatchMode=yes "$REMOTE" "$remote_script"

cat <<NEXT

==> Synced. Ask the user to run on orin11 (read the script first):
      less ~/qe-stage/query_engine/deploy/install.sh
      sudo bash ~/qe-stage/query_engine/deploy/install.sh ~/qe-stage/query_engine
NEXT
