# Meowsformer — FastAPI API 镜像
#
# 设计要点：
# - Debian bookworm + ffmpeg：与项目文档一致（转码、librosa 依赖）
# - libsndfile1：soundfile 读 WAV
# - CHROMA_DB_PATH=/data/chroma_db：可写持久化（Fly 可挂 volume 到 /data）
# - requirements-docker.txt：去掉 flet / sounddevice / zenodo-get 等仅本地或构建期工具
#
# 可选：构建时拉取 Zenodo 猫叫语料（体积大、耗时长），使流式匹配能读 wav：
#   docker build --build-arg FETCH_DATA=true -t meowsformer .
#
# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_DB_PATH=/data/chroma_db

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG FETCH_DATA=false
RUN mkdir -p /data/chroma_db \
    && if [ "$FETCH_DATA" = "true" ]; then \
         pip install --no-cache-dir zenodo-get \
         && python -m tools.download_datasets \
         && pip uninstall -y zenodo-get; \
       fi

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8080

# Fly.io 等平台注入 PORT；本地默认 8080
ENV PORT=8080
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
