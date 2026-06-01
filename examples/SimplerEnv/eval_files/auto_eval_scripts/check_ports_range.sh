#!/bin/bash
# Check port availability ahead of time to prepare for parallel testing
# Target port range
start_port=${START_PORT:-5400}
end_port=${END_PORT:-5500}
PORT_CHECK_PYTHON=${PORT_CHECK_PYTHON:-python3}

echo "Checking whether ports ${start_port}-${end_port} are available..."

for port in $(seq $start_port $end_port); do
  if "${PORT_CHECK_PYTHON}" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY
  then
    echo "Port $port is available"
  else
    echo "Port $port is in use"
  fi
done

echo "Port check complete!"
