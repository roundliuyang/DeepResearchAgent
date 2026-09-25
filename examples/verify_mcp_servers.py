"""验证 MCP 发现、TCP 注册和调用，不调用规划模型。

运行：python examples/verify_mcp_servers.py
"""

import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.config import config
from src.logger import logger
from src.tool import tcp


async def main():
    config.initialize(
        config_path=str(ROOT / "configs/bus.py"),
        args=Namespace(cfg_options={"workdir": "workdir/mcp_verification"}),
    )
    logger.initialize(config=config)
    try:
        await tcp.initialize(tool_names=["mcp_importer"])
        registered = [name for name in await tcp.list() if name != "mcp_importer"]
        assert set(registered) == {"mcp_local_math", "mcp_weather"}, registered
        print("Registered:", registered)

        math = await tcp(
            name="mcp_local_math", input={"action": "add", "args": {"a": 2, "b": 3}}
        )
        if not math.success:
            raise RuntimeError(math.message)
        content = json.loads(math.message)
        assert any(float(item["text"]) == 5 for item in content if item.get("type") == "text"), content
        print("STDIO add(2, 3):", math.message)

        weather = await tcp(
            name="mcp_weather",
            input={"action": "get_current_weather", "args": {"lat": 31.2304, "lon": 121.4737}},
        )
        if not weather.success:
            raise RuntimeError(weather.message)
        weather_blocks = json.loads(weather.message)
        weather_data = json.loads(next(item["text"] for item in weather_blocks if item.get("type") == "text"))
        assert isinstance(weather_data["stats"]["temperature_c"]["mean"], (int, float)), weather_data
        assert weather_data["period"], weather_data
        print("HTTP Shanghai weather:", weather.message)
        print("MCP verification passed")
    finally:
        await tcp.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
