"""测试 docx_server MCP 服务器

用法: 在项目根目录执行 python servers/demo_docx.py
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # 配置 docx_server 连接
    connections = {
        "docx_server": {
            "command": "python",
            "args": ["servers/docx_server.py"],
            "transport": "stdio"
        }
    }

    client = MultiServerMCPClient(connections, tool_name_prefix=False)

    # 获取工具列表
    tools = await client.get_tools(server_name="docx_server")
    print(f"发现 {len(tools)} 个工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 建立会话并调用 read_paragraphs
    async with client.session("docx_server") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools
        mcp_tools = await load_mcp_tools(session, server_name="docx_server", tool_name_prefix=False)

        read_tool = next(t for t in mcp_tools if t.name == "read_paragraphs")
        result = await read_tool.ainvoke({
            "file_path": "D:/资料/专利/一种面向空间复杂场景的通用智能体构建方法.docx"
        })
        print(f"\n=== 文档内容（前500字）===\n{result[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
