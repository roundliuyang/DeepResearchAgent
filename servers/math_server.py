"""
示例 MCP 数学服务器

提供基本的数学计算工具，演示如何创建 MCP 服务器。
"""

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("MathServer")


@mcp.tool()
def add(a: float, b: float) -> float:
    """两个数相加
    
    Args:
        a: 第一个数
        b: 第二个数
    
    Returns:
        两数之和
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """两个数相减
    
    Args:
        a: 被减数
        b: 减数
    
    Returns:
        两数之差
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """两个数相乘
    
    Args:
        a: 第一个数
        b: 第二个数
    
    Returns:
        两数之积
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """两个数相除
    
    Args:
        a: 被除数
        b: 除数（不能为0）
    
    Returns:
        两数之商
    
    Raises:
        ValueError: 当除数为0时
    """
    if b == 0:
        raise ValueError("除数不能为0")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """计算幂
    
    Args:
        base: 底数
        exponent: 指数
    
    Returns:
        base 的 exponent 次幂
    """
    return base ** exponent


if __name__ == "__main__":
    # 启动 MCP 服务器（stdio 传输）
    mcp.run()
