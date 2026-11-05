import webbrowser
import threading

import uvicorn

from .config import settings

if __name__ == "__main__":
    s = settings()
    url = f"http://{s['host']}:{s['port']}"
    print(f"\n  TM-Sim запущен: {url}\n  (закройте это окно, чтобы остановить программу)\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("tmsim.main:app", host=s["host"], port=s["port"], log_level="warning")
