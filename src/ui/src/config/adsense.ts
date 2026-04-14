/**
 * Google AdSense — 单一来源配置。
 *
 * `5131422115935645` 是发布商 ID（对应 ca-pub-5131422115935645）。
 * `data-ad-slot` 在控制台里多为另一串「广告单元 ID」；若线上无填充，请用各单元「获取代码」
 * 里的 data-ad-slot 替换下方默认值，或设置 VITE_ADSENSE_SLOT_* 环境变量（Docker/Fly 构建前注入）。
 */
export const ADSENSE_PUBLISHER_ID_NUMERIC = "5131422115935645";
export const VITE_ADSENSE_SLOT_RESULT = "9959506878";
export const VITE_ADSENSE_SLOT_ABOUT_MID = "5648952118";
export const VITE_ADSENSE_SLOT_ABOUT_BOTTOM = "9676435840";

export const ADSENSE_CLIENT_ID =
  `ca-pub-${ADSENSE_PUBLISHER_ID_NUMERIC}` as const;

function slotFromEnv(key: string): string | undefined {
  const v = import.meta.env[key];
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

/** 默认 slot：与发布商 ID 相同时 AdSense 可能不投放；请按需改为单元 ID 或设 VITE_ 变量。 */
const DEFAULT_DATA_AD_SLOT = ADSENSE_PUBLISHER_ID_NUMERIC;

/** 首页翻译结果下方 */
export const ADSENSE_SLOT_RESULT =
  slotFromEnv("VITE_ADSENSE_SLOT_RESULT") ?? DEFAULT_DATA_AD_SLOT;

/** 宽屏（xl+）主页与 Science 页左右侧栏；仍由 `VITE_ADSENSE_SLOT_ABOUT_MID` 注入 */
export const ADSENSE_SLOT_ABOUT_MID =
  slotFromEnv("VITE_ADSENSE_SLOT_ABOUT_MID") ?? DEFAULT_DATA_AD_SLOT;

/** Science 页底部 */
export const ADSENSE_SLOT_ABOUT_BOTTOM =
  slotFromEnv("VITE_ADSENSE_SLOT_ABOUT_BOTTOM") ?? DEFAULT_DATA_AD_SLOT;
