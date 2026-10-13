import argparse
import asyncio
import json
import os
import socket

import uvicorn

from .app import create_app
from .config import Config


async def serve(args):
    config = Config(demo=args.demo)
    app = create_app(config)
    sock = socket.socket()
    sock.bind(("127.0.0.1", args.port))
    sock.listen(8)
    sock.setblocking(False)
    print(json.dumps({"type": "server", "port": sock.getsockname()[1]}), flush=True)
    server = uvicorn.Server(
        uvicorn.Config(
            app, log_level="warning", access_log=False, ws_max_size=65536, ws_max_queue=32
        )
    )

    async def watch_parent():
        if not args.parent:
            return
        while not server.should_exit:
            await asyncio.sleep(2)
            if os.getppid() != args.parent:
                server.should_exit = True

    watcher = asyncio.create_task(watch_parent())
    try:
        await server.serve(sockets=[sock])
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent", type=int)
    parser.add_argument("--demo", action="store_true", help="Explicit test adapters; no real AI")
    asyncio.run(serve(parser.parse_args()))
