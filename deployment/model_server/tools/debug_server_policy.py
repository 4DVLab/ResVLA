"""
Debug / smoke-test client for deployment/model_server/server_policy.py.

Purpose:
  - Establish a WebSocket connection to the policy server.
  - Optionally run a ping or a simple inference request to verify end-to-end transport
    (serialization + server handling).

Usage example:
  python -m deployment.model_server.tools.debug_server_policy --host 127.0.0.1 --port 10093 --test infer

Notes:
  - The random observation is synthetic and only meant to validate the interface.
  - The inference request uses the current ResVLA predict_action schema: examples=[{"image": ..., "lang": ...}].
"""

import argparse
import logging
import numpy as np


from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WebSocket policy client smoke test (msgpack protocol)")
    ap.add_argument("--host", default="127.0.0.1", help="server hostname/IP (do not use 0.0.0.0)")
    ap.add_argument("--port", type=int, default=10093, help="server port")
    ap.add_argument("--api_key", default="", help="optional: API key for authentication")
    ap.add_argument(
        "--test", choices=["ping", "infer"], default="infer", help="test mode: ping transport only, or try inference"
    )
    ap.add_argument("--log_level", default="INFO")
    return ap


def _main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    ping_ret = client.predict_action({"type": "ping", "request_id": "smoke-test-ping"})
    logging.info("Ping resp: %s", ping_ret)

    if args.test == "infer":
        try:
            H, W = 224, 224
            img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

            request = {
                "type": "infer",
                "request_id": "smoke-test",
                "payload": {
                    "examples": [
                        {
                            "image": [img],
                            "lang": "debug: pick up the red block",
                        }
                    ],
                },
            }

            infer_ret = client.predict_action(request)
            logging.info("Infer resp: %s", infer_ret)
        except Exception as e:
            logging.error("Infer error (this still proves transport OK): %s", e)

    client.close()
    logging.info("Smoke test done.")


if __name__ == "__main__":
    _main()
