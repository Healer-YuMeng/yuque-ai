from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from scripts.build_index import main as build_main


if __name__ == "__main__":
    load_dotenv()
    os.environ.setdefault("BOOTSTRAP_QUERY", "知识库")
    asyncio.run(build_main())

