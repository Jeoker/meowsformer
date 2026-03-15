# Phase 7 Batch 1 — 解除服务器启动阻塞

| 属性 | 值 |
|------|-----|
| **状态** | ✅ 已完成 |
| **目标** | 移除所有 auth 导入，使 `python main.py` 正常启动、全部测试通过 |
| **优先级** | 最高（后续所有 Batch 的前提） |

---

## 完成范围

- `main.py` — 移除 auth 导入（L8-9）、`create_tables()` 调用（L36-40）、`auth_router` 注册（L65）
- `app/api/endpoints.py` — 移除 `get_optional_user` 导入（L4）
- `app/api/__init__.py` — 清空为空包（断开 `__init__.py` → `endpoints.py` → `app.auth` 连锁链）
- `app/core/config.py` — 移除 3 个 JWT 配置字段（Reviewer Warning 采纳）
- `src/ui/vite.config.ts` — 移除 `/auth` proxy（Reviewer Warning 采纳）
- `src/ui/src/App.vue` — 简化为占位页面，移除所有 auth 组件导入
- `tests/test_batch1_auth_removal.py` — 新增 21 个验证测试（6 个测试类）

---

## 修改文件清单

| 文件 | 操作 | 具体改动 |
|------|------|---------|
| `main.py` | 修改 | 移除 auth 导入 (L8-9)、`create_tables()` 调用块 (L36-40)、`auth_router` 注册 (L65) |
| `app/api/endpoints.py` | 修改 | 移除 `get_optional_user` 导入 (L4) |
| `app/api/__init__.py` | 修改 | 清空为空包，断开连锁导入链 |
| `app/core/config.py` | 修改 | 移除 3 个 JWT 配置字段 (Reviewer Warning 采纳) |
| `src/ui/vite.config.ts` | 修改 | 移除 `/auth` proxy (Reviewer Warning 采纳) |
| `src/ui/src/App.vue` | 修改 | 移除 auth 组件导入，简化为 TranslatePlaceholder 占位页面 |
| `tests/test_batch1_auth_removal.py` | 新增 | 21 个验证测试（导入完整性/路由/Health/配置/源码/Vite）|

---

## 验收结果

- `python main.py` 成功启动，`GET /health` 返回 200 ✅
- `python -m unittest discover tests` 全部 164 测试通过 ✅

---

## 测试新增

| 测试文件 | 用例数 | 描述 |
|----------|--------|------|
| `test_batch1_auth_removal.py` | 21 | 导入完整性、路由注册、Health 端点、配置无 JWT、源码无 auth 残留、Vite proxy |
