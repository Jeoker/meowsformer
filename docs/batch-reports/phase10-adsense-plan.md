# Phase 10 — 变现（AdSense / Ko-fi）

**工作流：** Sprint Mode（见 `.cursor/rules/dev-workflow.mdc`）  
**前置：** 公开 HTTPS 域名（生产站点）

**状态：** 进行中 — Batch 1 已完成；Batch 2–3 未开始。

---

## 1. 目标（为何是这三件事）

1. **内容页** — AdSense 需要站点有「有意义的内容」；纯单页工具站难过审。同一页面承担 **SEO 长尾**（如「猫叫含义」类检索），是流量的独立来源，不仅为过审。  
2. **Google AdSense** — 展示广告是 **流量 × RPM** 的可扩展层；接入前提是合规页面与可控版位。  
3. **Ko-fi** — **零审核成本** 的补充变现；可在 AdSense pending 期间并行，不替代广告策略。

**非目标：** Google Ads 广告主 API、Flet 内嵌广告、订阅/付费墙。

---

## 2. 执行顺序与时机

```
Batch 1：内容页 → Batch 2：广告与合规 → Batch 3：提交审核 + 推广
```

**推广须与提交审核尽量同一天启动：** 若推广过早，审核通过时高峰已过；若只审不推，通过前浪费等待。目标是 **审核通过时仍有流量可承接**。

---

## 3. 进度

| Batch | 内容 | 状态 |
|-------|------|------|
| 1 | 科学内容页 `/about`（审核与 SEO） | ✅ |
| 2 | `ads.txt`、手动广告单元、Ko-fi、`/privacy`、CMP | 待办 |
| 3 | 提交 AdSense 审核 + 推广（与审核同日） | 待办 |

---

## 4. Batch 1 — 记录（已完成）

**设计意图：** `/about` 用英文主文、科学溯源与 FAQ，同时满足 **审核对内容深度** 与 **搜索引擎可索引** 的需求。

**交付：** 英文页（约 600+ 词）；CatMeows / Meowsic、Zenodo DOI、五维标签与权重说明、场景 FAQ；顶栏 Demo / Science。

**改动：** `src/ui/src/router/index.ts`，`pages/AboutPage.vue`，`App.vue`，`main.ts`，`package.json`（`vue-router@4`）。

**验收：** `/about` 可访问；仓库无 `robots.txt` disallow，默认可爬取。

---

## 5. Batch 2 — 待办（要点说明）

**ads.txt** — 控制台注册站点、验证所有权后，将发布商 ID 写入 `src/ui/public/ads.txt`，构建后根路径可抓取。

**广告形态 — 用手动单元，不用 Auto ads：** Auto ads 会把广告插到任意位置，容易压在 **录音区 / 结果卡** 上，直接伤核心交互；改为在固定组件内放 **指定尺寸** 单元（例如结果卡下 300×250、About 文末 728×90），单份全局脚本 + `onMounted` push，路由切换不主动刷新单元（遵守政策）。

**Ko-fi** — Header 或结果区放官方按钮即可，无后端需求。

**隐私政策** — AdSense 要求可访问说明（Cookie、第三方广告数据）；可用生成器产出正文，挂 `/privacy` 并在页脚链出。

**EEA / CMP** — 部署在 Fly.io 即全球可达，**EEA 用户必然出现**，须在同意后再加载广告脚本；Cookiebot 等免费层或 Google 认证 CMP 均可，关键是 **顺序与同意状态** 与 AdSense 一致。

---

## 6. Batch 3 — 待办（要点说明）

- 控制台 **提交站点审核**（常见 1–14 天）；等待期 **禁止自行点击广告**（即使尚未展示）。  
- **与提交同日** 启动至少一条推广：Product Hunt、Reddit（如 r/cats、r/aww、r/InternetIsBeautiful）、短视频等，用演示素材换冷启动访客。

---

## 7. Phase 10 结束条件

1. `/about` 与导航 ✅  
2. `GET /ads.txt` → 200，内容正确  
3. 生产环境（无拦截）可见广告位、控制台无 AdSense 报错  
4. Ko-fi 可跳转  
5. `/privacy` 可达  
6. AdSense 已通过或审核中  
7. 至少完成一项推广发布  

---

## 8. 参考链接

- [AdSense 入门](https://support.google.com/adsense/answer/7402250) · [ads.txt](https://support.google.com/adsense/answer/7532444) · [手动单元](https://support.google.com/adsense/answer/9190028)  
- [EEA 同意](https://support.google.com/adsense/answer/13554116) · [Cookiebot 定价](https://www.cookiebot.com/en/pricing/) · [Ko-fi](https://ko-fi.com/) · [Product Hunt 发布](https://www.producthunt.com/posts/new)
