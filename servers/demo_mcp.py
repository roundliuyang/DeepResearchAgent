"""手动测试 MCP 服务器

用法: 在项目根目录执行 python servers/demo_mcp.py
注意: 不要通过 PyCharm 的 pytest 运行，async 函数需要 asyncio.run 驱动
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # 配置 MCP 服务器连接，args 路径相对于运行目录（项目根目录）
    connections = {
        "math_server": {
            "command": "python",
            "args": ["servers/math_server.py"],
            "transport": "stdio"
        }
    }

    # 创建多服务器客户端，tool_name_prefix=False 表示工具名不加服务器前缀
    # 配置里 "transport": "stdio"，MCP 客户端会启动 python servers/math_server.py 作为子进程，通过 stdin/stdout 管道通信，不走网络（端口）
    client = MultiServerMCPClient(connections, tool_name_prefix=False)

    # 第一步：获取指定服务器的工具列表（仅查看，不保持会话）
    tools = await client.get_tools(server_name="math_server")
    print(f"发现 {len(tools)} 个工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 第二步：建立持久会话并调用工具
    async with client.session("math_server") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools
        # 在会话中加载工具（此时工具可正常调用）
        mcp_tools = await load_mcp_tools(session, server_name="math_server", tool_name_prefix=False)

        # 测试加法
        add_tool = next(t for t in mcp_tools if t.name == "add")
        result = await add_tool.ainvoke({"a": 1, "b": 2})
        print(f"\n1 + 2 = {result}")

        # 测试乘法
        mul_tool = next(t for t in mcp_tools if t.name == "multiply")
        result = await mul_tool.ainvoke({"a": 3, "b": 4})
        print(f"3 * 4 = {result}")


if __name__ == "__main__":
    asyncio.run(main())
