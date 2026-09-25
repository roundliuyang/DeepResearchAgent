# MCP 服务接入

`configs/bus.py` 已配置两个服务，`tcp.initialize()` 内部自动完成 MCP 导入。
入口脚本只需调用原有的工具初始化方法，无需单独处理 MCP。
无需手动运行本地数学服务；MCP 客户端会启动并管理它的进程。

| 服务 | 传输方式 | TCP 名称 | 工具 |
| --- | --- | --- | --- |
| `local_math` | stdio，本地 Python 子进程 | `mcp_local_math` | add、subtract、multiply、divide、power |
| `weather` | Streamable HTTP | `mcp_weather` | get_current_weather、get_weather_forecast 等 |

本地服务在配置中显式指定 Python 解释器和脚本的绝对路径；
换机器或虚拟环境时修改这两项配置。
远程服务连接失败时，`run_bus.py` 会明确报告导入错误，不会假装工具已就绪。
如果不需要 MCP，可删除 `mcp_connections` 配置或设为空字典。

## 调用

每个 MCP 服务映射为一个 TCP 代理，使用 `action` 选择工具，`args` 传入参数：

```python
await tcp(name="mcp_local_math", input={
    "action": "add", "args": {"a": 2, "b": 3},
})
await tcp(name="mcp_weather", input={
    "action": "get_current_weather", "args": {"lat": 31.2304, "lon": 121.4737},
})
```

Agent 的工具上下文包含可用 action 名称。
可以将 `run_bus.py` 的任务改为：
“使用 mcp_local_math 计算 2+3，再使用 mcp_weather 查询上海天气，注明数据时间。”

## 不调用规划模型的验证

```bash
python examples/verify_mcp_servers.py
```

脚本实际验证服务发现、TCP 注册、加法结果和上海天气数据，
运行记录写在 `workdir/mcp_verification`；会消耗一次天气查询额度。
只依赖项目已有的 MCP SDK 和 `langchain_mcp_adapters` 环境。

## 免费天气服务

端点：`https://pixelgust.com/mcp`。
[PixelGust 官方说明](https://pixelgust.com/blog/weather-mcp-server) 提供免注册、
免 API Key 的免费访问，当前公开额度为每日 50 次调用，并非无限免费。
天气数据来自 GFS 等数据集，保留返回的 `period`，不要将其表述为实时地面观测。
额度、服务可用性和数据更新时间由第三方控制；规划模型的调用费用独立于 MCP 服务。
