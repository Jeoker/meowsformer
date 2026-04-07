# Docker 与 Fly.io 部署

本地开发见 [`development-overview.md`](./development-overview.md)「搭建与运行」。本文仅描述 **容器镜像** 与 **可选 Fly.io** 要点。

---

## Docker 镜像

| 文件 | 作用 |
|------|------|
| [`Dockerfile`](../Dockerfile) | 多阶段：`node:20` 构建 `src/ui` → Vite 产物拷至 `static/ui`；`python:3.12-slim-bookworm` 运行 FastAPI。 |
| [`requirements-docker.txt`](../requirements-docker.txt) | 运行时依赖子集：无 Flet / flet-audio / sounddevice / zenodo-get（与桌面端、数据下载 CLI 解耦）。 |
| [`.dockerignore`](../.dockerignore) | 减小 context：忽略 `src/ui/node_modules`、`db/chroma_db`、`assets/raw_data`、宿主 `requirements.txt` 等。 |

**环境变量**

- `CHROMA_DB_PATH=/data/chroma_db`（镜像内默认）：Chroma 持久目录；与本地 `./db/chroma_db` 分离。
- `PORT`：由 Fly 等平台注入；[`CMD`](../Dockerfile) 为 `uvicorn main:app --host 0.0.0.0 --port ${PORT}`。

**系统包（镜像内）**：`ffmpeg`、`libsndfile1`、`libgomp1`（与转码 / librosa / soundfile 一致）。

**可选构建参数 `FETCH_DATA=true`**：构建阶段执行 `python -m tools.download_datasets`（会先 `pip install zenodo-get`），把 Zenodo 语料拉入镜像，使流式管线能读 `assets/raw_data/...` 下 wav。**构建慢、镜像大。**

**将已有 Chroma 数据打进镜像**

1. 在本机用有效 `OPENAI_API_KEY` 跑通服务，使 `db/chroma_db` 已生成并含数据。
2. 构建前允许把 `db/chroma_db` 纳入 context：在 `.dockerignore` 中**去掉或注释** `db/chroma_db` 一行。
3. 在 `Dockerfile` 中于 `COPY . .` 之后增加：`COPY db/chroma_db/ /data/chroma_db/`（路径与 `CHROMA_DB_PATH` 一致）。
4. `docker build` / `fly deploy`。

若不做 2–3，空库可在首次启动时由 `initialize_knowledge_base()` 写入，但需运行时能调 embedding API。

---

## Fly.io

**CLI**：官方安装脚本 [Install flyctl](https://fly.io/docs/hands-on/install-flyctl/)，`fly auth login`。

**与本仓库对齐**

- [`fly.toml`](../fly.toml)：`internal_port = 8080`，与镜像 `EXPOSE 8080` / `PORT` 一致。
- 密钥勿写入镜像：`fly secrets set OPENAI_API_KEY=...`；若用 ai-builders：`API_PROVIDER`、`AI_BUILDER_TOKEN`、`AI_BUILDER_BASE_URL`（见 `app/core/config.py`）。

**常用命令**

```bash
fly deploy
fly status
fly logs
```

**验收**：`GET /health`、`GET /docs`；若已构建前端进镜像，根路径 `/` 为 Vue SPA。

---

## 计费与文档（官方）

Fly 按组织计量；**没有单独的「Docker 镜像仓存储费」条目**，与容器相关的磁盘在文档中多表述为 Machine 的 **rootfs**（OCI 镜像 + 运行时层）。详见：

- [Resource Pricing](https://fly.io/docs/about/pricing/)（Compute、Stopped Machines 与 rootfs、Volumes、出站流量等）
- [Machine billing](https://fly.io/docs/about/billing/#machine-billing)（运行中按 CPU/RAM；停止/挂起时 rootfs 另计）
- [Free Trial](https://fly.io/docs/about/free-trial/)（试用额度与绑卡后计费关系以官网为准）
- [Cost Management](https://fly.io/docs/about/cost-management/)（Volume、扩展服务、独立 IPv4 等易忽略项）

**降低或停止费用**：删除不用的 App、Volume；控制台检查 Managed Postgres / Redis / Tigris 等扩展；详见 Cost Management 与 Billing 页。
