"""Example client to test the serverless endpoint.

Usage:
    export RUNPOD_ENDPOINT_ID=xxxxxxxx
    export RUNPOD_API_KEY=xxxxxxxx
    python test_request.py path/to/invoice.pdf
"""
import base64
import json
import os
import sys
import time

import httpx


def main():
    if len(sys.argv) < 2:
        print("usage: python test_request.py <path/to/invoice.pdf>")
        sys.exit(1)

    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not endpoint_id or not api_key:
        print("set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY env vars")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        file_b64 = base64.b64encode(f.read()).decode()

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "file": file_b64,
            "filename": os.path.basename(path),
            "review_threshold": 0.85,
            "request_id": f"test-{int(time.time())}",
        }
    }

    t0 = time.time()
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(url, headers=headers, json=payload)
    elapsed = time.time() - t0

    print(f"HTTP {resp.status_code} in {elapsed:.1f}s")
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
