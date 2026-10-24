"""打包入口：启动 Web UI（PyInstaller 用）。"""
import sys

from accs.web import main

if __name__ == "__main__":
    sys.exit(main())
