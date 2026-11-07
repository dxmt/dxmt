# 用 DXMT 作为 MaxKB 智能体的记忆（MCP）

MaxKB 的 AI 对话节点和 MCP 工具可以通过 Streamable HTTP 调用 MCP 服务器。DXMT 在 `POST /mcp`
提供 MCP 服务，所以 MaxKB 智能体不需要任何代码就能在 DXMT 里存取记忆：`remember`、`recall`、
`forget`、`list_memories`；服务器配置了记忆模型时，还有 `extract_memories`（从对话中提炼事实）
和 `update_profile`（生成人物画像）。

它给 MaxKB 带来什么：MaxKB 自带的长期记忆是每个应用、每个用户一段文本，由模型整段重写、整段
塞进 prompt。接上 DXMT 后，智能体只召回当前问题需要的那几条（按语义和关键词），而且同一份记忆
可以被服务器上的其他智能体共用：另一个 MaxKB 应用、Hermes、Claude Code。

## 1. 把 DXMT 部署在 MaxKB 能访问到的地方

**最容易出错的就是这一步。** MaxKB 官方镜像里的 MCP 连接运行在沙箱中，沙箱会拒绝连接
`MAXKB_SANDBOX_PYTHON_BANNED_HOSTS` 里的地址，默认值是：

    127.0.0.0/8, localhost, host.docker.internal, 172.17.0.0/16, maxkb, pgsql, redis,
    172.31.250.192/26, 0.0.0.0/32, ::/0（以及 MaxKB 容器自己的 IP）

所以 `http://localhost:9999/mcp`、`http://host.docker.internal:9999/mcp` 以及 Docker 默认网桥上的
地址都会失败，哪怕服务器本身能访问。推荐把 DXMT 和 MaxKB 放在同一个 Docker 网络里，用服务名访问：

```yaml
# docker-compose.yml
services:
  maxkb:
    image: 1panel/maxkb:latest
    ports: ["8080:8080"]
    volumes: ["maxkb:/opt/maxkb"]
    networks: [ai]
  dxmt:
    image: dxmtai/dxmt:latest
    environment:
      DXMT_EMBEDDING_BASE_URL: http://your-model-server:1234/v1   # 任意 OpenAI 兼容的 /embeddings
      DXMT_EMBEDDING_MODEL: bge-m3
      # DXMT_API_KEY: change-me                                  # 设置后需要在 MaxKB 里带上，见下文
      # DXMT_MEMORY_LLM_MODEL: qwen3                             # 开启 extract_memories / update_profile
    volumes: ["dxmt:/data"]
    networks: [ai]
networks:
  ai: {}
volumes:
  maxkb: {}
  dxmt: {}
```

这样 MaxKB 用 `http://dxmt:9999/mcp` 访问 DXMT。部署在另一台机器上的 DXMT，只要地址不在禁止列表里，
也一样可以用。也可以在 MaxKB 容器上改 `MAXKB_SANDBOX_PYTHON_BANNED_HOSTS` 来放开，但这会削弱 MaxKB
有意设置的保护，优先用共享网络的方式。

## 2. 在 MaxKB 里添加 MCP 服务器

在 AI 对话节点的 MCP 设置里（或在"工具"里新建 MCP 工具）填入：

```json
{
  "dxmt": {
    "transport": "streamable_http",
    "url": "http://dxmt:9999/mcp"
  }
}
```

如果服务器设置了 `DXMT_API_KEY`，加上 `"headers": {"api-key": "change-me"}`（或
`"Authorization": "Bearer change-me"`）。

## 3. 在提示词里告诉模型什么时候用

工具自带说明，但只有提示词明确要求，模型才会稳定地使用记忆。可以参考这样的系统提示词：

> 每次对话开始时，先用用户的问题调用 `recall`，看看你已经知道关于他的什么。当用户说出长期有效的
> 信息（偏好、关于自己的事实、做出的决定）时，调用 `remember` 记下来，kind 用 `preference` 或
> `fact`。每条记忆都打上 `user:<用户 id>` 标签，`recall` 时也带上同一个标签，这样不同用户互相看不到。

标签里的用户 id，可以在提示词里插入 MaxKB 的用户变量（开始节点的 `chat_user_id`）。服务器配置了
`DXMT_MEMORY_LLM_MODEL` 时，可以在对话结束的环节把对话记录交给 `extract_memories`，而不是依赖模型
一条条地记；再用 `update_profile` 维护人物画像，之后用 kind 为 `profile` 的 `recall` 读出来。

## 测试情况

2026-11-07，用官方 `1panel/maxkb:latest` 镜像（MaxKB v2.10.8-lts，沙箱开启）和
`dxmtai/dxmt:latest`（1.6.0），放在同一个 Docker 网络里：通过 MaxKB 自己的 MCP 客户端
（`common.mcp.client.create_mcp_client`，也就是对话节点实际走的、经过沙箱的那条路径），对
`http://dxmt:9999/mcp` 列出了 DXMT 的工具，`remember` 和 `recall` 都正常。同样的调用换成 DXMT 在
默认网桥上的地址，在沙箱里失败了，而同一个容器里直接 `curl` 能访问到：这就是禁止列表在起作用。
这次验证走的是这个客户端，没有在 MaxKB 界面上逐步点击。`extract_memories` 和 `update_profile`
是 1.6.0 之后加的，用同一个客户端库（`langchain-mcp-adapters` 0.3.0、`mcp` 1.30.0）对新版服务器测过。
