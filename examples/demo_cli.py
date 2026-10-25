"""Demonstrate calling the Git CLI through BashTool."""

import asyncio
import sys
from pathlib import Path

# Make src importable when running this script directly.
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.tool.default_tools.bash import BashTool


async def main():
    tool = BashTool()
    command = "git --version"

    print(f"执行命令：{command}")
    result = await tool(command=command)
    print(result.message)


if __name__ == "__main__":
    asyncio.run(main())
