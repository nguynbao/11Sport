#!/usr/bin/env bash
# Generate an English speech test video (uses macOS `say`).
set -euo pipefail
OUT=${1:-sample.mp4}
FFMPEG=${FFMPEG_BIN:-$(command -v ffmpeg || echo "$HOME/.local/bin/ffmpeg")}
say -v Samantha -o /tmp/speech.aiff "Hello everyone. Today we introduce our new running shoes. They are light, durable and very comfortable. Thank you for watching."
"$FFMPEG" -y -f lavfi -i testsrc=size=1280x720:rate=25 -i /tmp/speech.aiff \
  -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac "$OUT"
echo "Created $OUT"
