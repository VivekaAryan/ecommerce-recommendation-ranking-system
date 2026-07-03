"""Uvicorn entrypoint for the testing dashboard."""

from __future__ import annotations

import argparse

from recsys.env import configure_runtime_env

configure_runtime_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recsys testing dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "recsys.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
