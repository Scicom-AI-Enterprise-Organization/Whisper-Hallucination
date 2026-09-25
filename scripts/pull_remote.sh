#!/bin/bash
# Copy one file back from the GPU box.
#
#   bash scripts/pull_remote.sh bench/scores.json
#
# `scp` fails while claude-ping's master connection is up, and `sync` only goes one way
# (laptop -> box) with --delete, so anything generated remotely has to come back explicitly.
# base64 rather than cat: the transport prefixes its own lines on retry, and a stray
# "[claude-ping] transient failure" inside a JSON file is silent corruption.
set -eu
CP=${CP:-/Users/husein.z/Documents/claude-ping/claude-ping}
REMOTE=${1:?usage: pull_remote.sh <path relative to remote_dir> [local path]}
LOCAL=${2:-$REMOTE}

$CP exec "cd /root/whisper-halluc-train && base64 -w0 '$REMOTE'" \
  | grep -v '^\[claude-ping\]' | tr -d '\n' | base64 -d > "$LOCAL.part"
mv "$LOCAL.part" "$LOCAL"
echo "-> $LOCAL  ($(wc -c < "$LOCAL") bytes)"
