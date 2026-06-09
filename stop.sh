#!/bin/bash
# Stop the aimodel proxy and (optionally) Ollama.
set -e

PIDS_FILE="/tmp/aimodel.pids"

if [ ! -f "$PIDS_FILE" ]; then
  echo "aimodel does not appear to be running (no $PIDS_FILE found)."
  echo "To kill manually:"
  echo "  pkill -f 'aimodel/proxy/server.py'"
  echo "  pkill -x ollama"
  exit 0
fi

# shellcheck disable=SC1090
source "$PIDS_FILE"

if [ -n "${PROXY_PID:-}" ]; then
  if kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Stopping proxy (PID $PROXY_PID)..."
    kill "$PROXY_PID"
  else
    echo "Proxy (PID $PROXY_PID) already stopped."
  fi
fi

if [ -n "${OLLAMA_PID:-}" ]; then
  if kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "Stopping Ollama (PID $OLLAMA_PID)..."
    kill "$OLLAMA_PID"
  else
    echo "Ollama (PID $OLLAMA_PID) already stopped."
  fi
fi

rm -f "$PIDS_FILE"
echo "Done."
