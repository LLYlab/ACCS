from __future__ import annotations

import argparse
import os
import sys

from .webapp import app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="accs-web", description="ACCS Web UI(实时识别 + 翻译)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="开发模式,代码变更自动重载")
    args = p.parse_args(argv)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
