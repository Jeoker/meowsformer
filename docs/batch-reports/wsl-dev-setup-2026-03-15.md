# WSL2 开发环境验证（2026-03-15）

| 字段 | 值 |
|------|-----|
| **状态** | ✅ 已完成 |
| **目标** | WSL2 + WSLg 下打通 Flet 桌面模式与麦克风录音 |
| **日期** | 2026-03-15 |

---

## 决策

取消 Batch 6（Web 浏览器兼容）：预构建 Web 客户端不包含 flet-audio 扩展，方案不可行。改用 WSLg 桌面模式开发。

---

## 推荐启动命令

```bash
# 终端 1
python main.py

# 终端 2（WSL2 需先完成 [wsl2-audio-setup.md](../wsl2-audio-setup.md)）
MEOWSFORMER_FLET_VIEW=desktop flet run -m src.flet_mobile.app
```

> 必须用 `flet run -m` 而非 `python -m`，否则预构建客户端不包含 flet-audio，会显示 "Unknown control: Audio"。
