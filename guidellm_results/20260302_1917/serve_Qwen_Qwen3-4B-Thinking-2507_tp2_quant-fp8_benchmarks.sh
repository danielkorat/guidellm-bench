#!/bin/bash
# Auto-generated – serves the HTML report and opens it in your browser.
# SCRIPT_DIR is the absolute path baked in at generation time — cwd-independent.
SCRIPT_DIR="/root/guidellm-bench/guidellm_results/20260302_1917"
PORT=8081
HTML="Qwen_Qwen3-4B-Thinking-2507_tp2_quant-fp8_benchmarks.html"
# Kill any process already listening on $PORT
OLD_PID=$(lsof -ti tcp:$PORT 2>/dev/null)
if [ -n "$OLD_PID" ]; then
    echo "Killing existing server on port $PORT (PID $OLD_PID)"
    kill "$OLD_PID" 2>/dev/null
    sleep 1
fi
echo "Serving http://localhost:$PORT/$HTML  (Ctrl-C to stop)"
python3 -m http.server "$PORT" --directory "$SCRIPT_DIR" &
SERVER_PID=$!
sleep 1
"$BROWSER" "http://localhost:$PORT/$HTML" 2>/dev/null || xdg-open "http://localhost:$PORT/$HTML" 2>/dev/null || open "http://localhost:$PORT/$HTML" 2>/dev/null || echo "Open http://localhost:$PORT/$HTML in your browser"
wait "$SERVER_PID"
