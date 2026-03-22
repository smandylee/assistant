from livekit.agents import WorkerOptions, cli

from faust_app import entrypoint
from faust_env import setup_env

if __name__ == "__main__":
    setup_env()
    # 워커 실행
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))