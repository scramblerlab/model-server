#!/bin/bash
# Start the unified aimodel LLM server (Ollama + stats proxy).
# Run from the project root:  ./start.sh [--model <name>]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# ── Load .env ──────────────────────────────────────────────────────────────
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +o allexport
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:4b}"
AIMODEL_OLLAMA_PORT="${AIMODEL_OLLAMA_PORT:-11434}"
AIMODEL_PROXY_PORT_RADIO="${AIMODEL_PROXY_PORT_RADIO:-11430}"
AIMODEL_PROXY_PORT_LOGGER="${AIMODEL_PROXY_PORT_LOGGER:-11431}"

# ── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--model <name>]"
      exit 1
      ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║        aimodel — Starting LLM Server         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Ollama install / update check ──────────────────────────────────────
echo "[1/4] Checking Ollama..."

if ! command -v ollama &>/dev/null; then
  echo "  Ollama not found — installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "  Ollama installed."
else
  # NOTE: `ollama --version` reports the version of any *running server* (which
  # may be a different install, e.g. the menu-bar app), so query the package
  # manager for the CLI's own version when Homebrew manages it.
  OLLAMA_VIA_BREW=""
  if command -v brew &>/dev/null && brew list ollama &>/dev/null; then
    OLLAMA_VIA_BREW=1
    INSTALLED_VERSION=$(brew list --versions ollama 2>/dev/null \
      | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
  else
    INSTALLED_VERSION=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
  fi
  LATEST_VERSION=$(curl -fsSL "https://api.github.com/repos/ollama/ollama/releases/latest" 2>/dev/null \
    | grep '"tag_name"' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")

  if [ "$LATEST_VERSION" = "unknown" ] || [ "$INSTALLED_VERSION" = "unknown" ]; then
    echo "  Ollama $INSTALLED_VERSION (version check skipped — no internet or rate-limited)"
  elif [ "$INSTALLED_VERSION" != "$LATEST_VERSION" ]; then
    echo "  Ollama $INSTALLED_VERSION installed; $LATEST_VERSION available."
    echo "  Updating Ollama..."
    if [ -n "$OLLAMA_VIA_BREW" ]; then
      brew upgrade ollama
      echo "  Ollama updated to $LATEST_VERSION."
    else
      # The CLI is the Ollama.app bundle (ollama.com/install.sh is Linux-only,
      # so it cannot update a macOS install). The app self-updates on launch;
      # a transient mismatch here is harmless — just inform and continue.
      echo "  NOTE: Ollama is the macOS app bundle — open Ollama.app once to let it self-update."
    fi
  else
    echo "  Ollama $INSTALLED_VERSION (up to date)"
  fi
fi

# ── 2. Start Ollama ────────────────────────────────────────────────────────
echo ""
echo "[2/4] Starting Ollama (port $AIMODEL_OLLAMA_PORT)..."

OLLAMA_PID=""
export OLLAMA_HOST="127.0.0.1:$AIMODEL_OLLAMA_PORT"

# Performance tuning for Apple Silicon (M4 Pro, 64 GB).
# These are exported so the ollama subprocess inherits them.
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_KEEP_ALIVE=-1
# Note: the variable is OLLAMA_CONTEXT_LENGTH — OLLAMA_NUM_CTX does not exist
# and is silently ignored by Ollama (the model then loads at its native max
# context, e.g. 262K for qwen3.5 → ~9 GB of extra KV cache).
export OLLAMA_CONTEXT_LENGTH=8192

# If Ollama is already running, reuse it only when a previous aimodel run
# started it (recorded PID still alive). Anything else — typically the Ollama
# menu-bar app launched at login — is missing the performance env vars above,
# so stop it and start our own instance.
REUSE_OLLAMA=""
if curl -sf "http://127.0.0.1:$AIMODEL_OLLAMA_PORT/api/tags" &>/dev/null; then
  RECORDED_PID=$(grep -E '^OLLAMA_PID=' /tmp/aimodel.pids 2>/dev/null | cut -d= -f2)
  if [ -n "$RECORDED_PID" ] && kill -0 "$RECORDED_PID" 2>/dev/null; then
    echo "  Ollama already running (aimodel-managed, PID $RECORDED_PID)."
    REUSE_OLLAMA=1
    OLLAMA_PID="$RECORDED_PID"
  else
    echo "  Ollama is running but was NOT started by aimodel (menu-bar app or manual launch)"
    echo "  — its instance ignores the performance settings above. Restarting it..."
    osascript -e 'quit app "Ollama"' &>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    WAIT=0
    while curl -sf "http://127.0.0.1:$AIMODEL_OLLAMA_PORT/api/tags" &>/dev/null; do
      sleep 1
      WAIT=$((WAIT + 1))
      if [ $WAIT -ge 15 ]; then
        echo "  ERROR: could not stop the existing Ollama instance on port $AIMODEL_OLLAMA_PORT."
        echo "  Quit the Ollama menu-bar app manually, then re-run this script."
        exit 1
      fi
    done
    echo "  Existing Ollama stopped."
  fi
fi

if [ -z "$REUSE_OLLAMA" ]; then
  echo "  Launching Ollama..."
  ollama serve > /tmp/aimodel-ollama.log 2>&1 &
  OLLAMA_PID=$!
  echo "  Ollama PID: $OLLAMA_PID  (log: /tmp/aimodel-ollama.log)"

  WAIT=0
  until curl -sf "http://127.0.0.1:$AIMODEL_OLLAMA_PORT/api/tags" &>/dev/null; do
    sleep 1
    WAIT=$((WAIT + 1))
    if [ $WAIT -ge 30 ]; then
      echo ""
      echo "  ERROR: Ollama did not start within 30s."
      echo "  Check the log: /tmp/aimodel-ollama.log"
      exit 1
    fi
  done
  echo "  Ollama ready (${WAIT}s)."
fi

# ── 3. Ensure model is available ───────────────────────────────────────────
echo ""
echo "[3/4] Checking model: $OLLAMA_MODEL..."

if OLLAMA_HOST="127.0.0.1:$AIMODEL_OLLAMA_PORT" ollama list 2>/dev/null \
    | grep -q "^${OLLAMA_MODEL}"; then
  echo "  Model $OLLAMA_MODEL already present."
else
  echo "  Pulling $OLLAMA_MODEL..."
  OLLAMA_HOST="127.0.0.1:$AIMODEL_OLLAMA_PORT" ollama pull "$OLLAMA_MODEL"
  echo "  Model $OLLAMA_MODEL ready."
fi

# ── 4. Start proxy ─────────────────────────────────────────────────────────
echo ""
echo "[4/4] Starting aimodel proxy..."

PROXY_DIR="$SCRIPT_DIR/proxy"
VENV="$PROXY_DIR/.venv"

if [ ! -f "$VENV/bin/python" ]; then
  echo "  Creating Python venv..."
  python3 -m venv "$VENV"
fi

# Install / update dependencies if requirements changed.
"$VENV/bin/pip" install -q -r "$PROXY_DIR/requirements.txt"

AIMODEL_OLLAMA_URL="http://127.0.0.1:$AIMODEL_OLLAMA_PORT" \
AIMODEL_PROXY_PORT_RADIO="$AIMODEL_PROXY_PORT_RADIO" \
AIMODEL_PROXY_PORT_LOGGER="$AIMODEL_PROXY_PORT_LOGGER" \
  "$VENV/bin/python" "$PROXY_DIR/server.py" &
PROXY_PID=$!
echo "  Proxy PID: $PROXY_PID"

# Wait for both proxy ports to respond.
for PORT in "$AIMODEL_PROXY_PORT_RADIO" "$AIMODEL_PROXY_PORT_LOGGER"; do
  WAIT=0
  until curl -sf "http://127.0.0.1:$PORT/health" &>/dev/null; do
    sleep 0.5
    WAIT=$((WAIT + 1))
    if [ $WAIT -ge 20 ]; then
      echo "  ERROR: Proxy did not bind on port $PORT within 10s."
      kill "$PROXY_PID" 2>/dev/null
      exit 1
    fi
  done
done
echo "  Proxy ready."

# ── macOS GPU memory ceiling info ─────────────────────────────────────────
WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "unavailable")
TOTAL_MB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
if [ "$WIRED_MB" != "unavailable" ] && [ "$TOTAL_MB" -gt 0 ]; then
  PCT=$(( WIRED_MB * 100 / TOTAL_MB ))
  RECOMMENDED_MB=$(( TOTAL_MB * 90 / 100 ))
  GPU_INFO="$WIRED_MB MB ($PCT% of ${TOTAL_MB} MB)"
  GPU_CMD="sudo sysctl iogpu.wired_limit_mb=$RECOMMENDED_MB"
else
  GPU_INFO="unavailable"
  GPU_CMD="sudo sysctl iogpu.wired_limit_mb=<90% of your RAM in MB>"
fi

# ── Startup banner ─────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║           aimodel — LLM Server Ready                 ║"
echo "╠═══════════════════════════════════════════════════════╣"
printf "║  Model:    %-43s║\n" "$OLLAMA_MODEL"
printf "║  Ollama:   http://127.0.0.1:%-26s║\n" "${AIMODEL_OLLAMA_PORT}"
printf "║  radio  →  http://127.0.0.1:%-26s║\n" "${AIMODEL_PROXY_PORT_RADIO}"
printf "║  logger →  http://127.0.0.1:%-26s║\n" "${AIMODEL_PROXY_PORT_LOGGER}"
echo "╠═══════════════════════════════════════════════════════╣"
printf "║  GPU memory ceiling: %-33s║\n" "$GPU_INFO"
if [ "$PCT" != "" ] && [ "$PCT" -lt 85 ]; then
printf "║  To extend to 90%%:                                    ║\n"
printf "║    %-51s║\n" "$GPU_CMD"
fi
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Logs:                                                ║"
echo "║    Ollama:  /tmp/aimodel-ollama.log                   ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Write PIDs for stop.sh
echo "PROXY_PID=$PROXY_PID" > /tmp/aimodel.pids
[ -n "$OLLAMA_PID" ] && echo "OLLAMA_PID=$OLLAMA_PID" >> /tmp/aimodel.pids

echo "Press Ctrl+C to stop."
echo ""

cleanup() {
  echo ""
  echo "Shutting down aimodel..."
  kill "$PROXY_PID" 2>/dev/null || true
  [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null || true
  rm -f /tmp/aimodel.pids
  echo "Done."
  exit 0
}
trap cleanup INT TERM

wait
