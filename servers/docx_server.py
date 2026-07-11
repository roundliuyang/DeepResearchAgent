"""
Word 文档解析 MCP 服务器

提供读取 Word 文档段落和表格内容的工具。
"""

from mcp.server.fastmcp import FastMCP
from docx import Document

mcp = FastMCP("DocxServer")


@mcp.tool()
def read_paragraphs(file_path: str) -> str:
    """读取 Word 文档的全部段落文本

    Args:
        file_path: docx 文件的绝对路径

    Returns:
        所有非空段落文本，每段一行
    """
    doc = Document(file_path)
    lines = []
    for idx, para in enumerate(doc.paragraphs):
        if para.text.strip():
            lines.append(f"[P{idx}] {para.text}")
    return "\n".join(lines) if lines else "（文档无段落内容）"


if __name__ == "__main__":
    mcp.run()
