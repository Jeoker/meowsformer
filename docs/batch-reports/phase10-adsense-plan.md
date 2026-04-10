# Phase 10 — 变现（AdSense / Ko-fi）

**工作流：** Sprint Mode（见 `.cursor/rules/dev-workflow.mdc`）  
**前置：** 公开 HTTPS 域名（生产站点）

**状态：** 进行中 — Batch 1–2 已完成；Batch 3 未开始。

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
| 2 | `ads.txt`、手动广告单元、Ko-fi、`/privacy`、CMP | ✅ |
| 3 | 提交 AdSense 审核 + 推广（与审核同日） | 待办 |

---

## 4. Batch 1 — 记录（已完成）

**设计意图：** `/about` 用英文主文、科学溯源与 FAQ，同时满足 **审核对内容深度** 与 **搜索引擎可索引** 的需求。

**交付：** 英文页（约 600+ 词）；CatMeows / Meowsic、Zenodo DOI、五维标签与权重说明、场景 FAQ；顶栏 Demo / Science。

**改动：** `src/ui/src/router/index.ts`，`pages/AboutPage.vue`，`App.vue`，`main.ts`，`package.json`（`vue-router@4`）。

**验收：** `/about` 可访问；仓库无 `robots.txt` disallow，默认可爬取。

---

## 5. Batch 2 — 记录（已完成）

**设计意图：** 3 个手动广告单元（不用 Auto ads），CMP 同意横幅控制 AdSense 脚本加载顺序，Ko-fi 导航栏链接，`/privacy` 隐私政策页。

**广告位设计（3 个手动单元）：**

| # | 页面 | 位置 | 占位 Slot ID | 理由 |
|---|------|------|-------------|------|
| Ad-1 | `/` TranslatePage | ResultSection 下方、footer 上方 | `AD_SLOT_RESULT` | 交互完成后的自然停顿点；`v-if="result"` 条件渲染，首次访问零广告 |
| Ad-2 | `/about` AboutPage | Meowsic 与 Five Dimensions 节之间 | `AD_SLOT_ABOUT_MID` | 研究背景→技术细节的内容转折间隙 |
| Ad-3 | `/about` AboutPage | FAQ 之后、disclaimer footer 之前 | `AD_SLOT_ABOUT_BOTTOM` | 经典文末广告位，深度阅读用户价值最高 |

**交付清单：**

| 类别 | 文件 | 说明 |
|------|------|------|
| 公共资源 | `src/ui/public/ads.txt` | 占位 publisher ID `ca-pub-XXXXXXXXXXXXXXXX` |
| 同意管理 | `src/ui/src/composables/useConsent.ts` | 单例 consent 状态 + AdSense 脚本条件加载 |
| 广告组件 | `src/ui/src/components/ads/AdUnit.vue` | consent accepted 才渲染 `<ins>`；`onMounted` push |
| 同意横幅 | `src/ui/src/components/ads/ConsentBanner.vue` | fixed bottom，Accept/Decline，仅 pending 时显示 |
| 隐私政策 | `src/ui/src/pages/PrivacyPage.vue` | 5 章节英文（数据收集、Cookie、第三方、权利、联系） |
| 路由 | `src/ui/src/router/index.ts` | 新增 `/privacy` |
| 全局布局 | `src/ui/src/App.vue` | +ConsentBanner、+Ko-fi nav link、+全局 footer（Privacy Policy · ©） |
| 翻译页 | `src/ui/src/pages/TranslatePage.vue` | +Ad-1（条件渲染） |
| 内容页 | `src/ui/src/pages/AboutPage.vue` | +Ad-2（文中）、+Ad-3（文末） |
| 类型声明 | `src/ui/src/env.d.ts` | `Window.adsbygoogle` 全局类型 |

**验收：** `npm run build` 通过（vue-tsc + Vite）；`python -m unittest discover tests` 331 个测试全部通过。

**上线前须替换的占位符：**
- `ca-pub-XXXXXXXXXXXXXXXX` → 真实 AdSense publisher ID（`useConsent.ts` + `ads.txt`）
- `AD_SLOT_RESULT` / `AD_SLOT_ABOUT_MID` / `AD_SLOT_ABOUT_BOTTOM` → AdSense 控制台创建的广告单元 ID
- `https://ko-fi.com/PLACEHOLDER` → 真实 Ko-fi 页面 URL
- `privacy@meowsformer.example` → 真实联系邮箱

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
