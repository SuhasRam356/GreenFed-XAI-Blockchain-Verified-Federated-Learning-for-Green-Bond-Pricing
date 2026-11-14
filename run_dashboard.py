"""
GreenFed-XAI Enterprise Dashboard Launcher
==========================================
Launches both the FastAPI REST backend (port 8000) and the Next.js 14
interactive web dashboard (port 3000) concurrently.
"""

import os
import sys
import subprocess
import time
import webbrowser

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")


def main():
    print("=" * 70)
    print("  GreenFed-XAI Enterprise Dashboard Launch Orchestrator")
    print("=" * 70)
    print("1. Starting FastAPI Backend (http://127.0.0.1:8000)...")

    # Start FastAPI
    api_cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"]
    api_proc = subprocess.Popen(api_cmd, cwd=ROOT_DIR)

    time.sleep(2)

    print("2. Starting Next.js 14 Frontend (http://localhost:3000)...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    front_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=FRONTEND_DIR)

    print("\n[SUCCESS] Both services are running!")
    print("  -> Frontend UI:   http://localhost:3000")
    print("  -> REST API Docs: http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        time.sleep(3)
        webbrowser.open("http://localhost:3000")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down GreenFed-XAI services...")
        api_proc.terminate()
        front_proc.terminate()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
