"""Start the Retrodict web page.

  python scripts/serve.py          -> open http://127.0.0.1:8000
  python scripts/serve.py --lan    -> also open it from a phone on the same Wi-Fi (http://<this PC's IP>:8000)
"""
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    lan = "--lan" in sys.argv
    host = "0.0.0.0" if lan else "127.0.0.1"
    print("Retrodict: http://127.0.0.1:8000" + (f"  (phone on the same Wi-Fi: http://{lan_ip()}:8000)" if lan else ""))
    uvicorn.run("app.web:app", host=host, port=8000, log_level="warning")
