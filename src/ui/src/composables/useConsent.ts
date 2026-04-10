import { ref } from "vue";

export type AdConsent = "pending" | "accepted" | "declined";

export const ADSENSE_CLIENT_ID = "ca-pub-5131422115935645";

const STORAGE_KEY = "ad-consent";

const SCRIPT_ATTR = "data-meowsformer-adsense";

function readInitial(): AdConsent {
  if (typeof localStorage === "undefined") return "pending";
  const v = localStorage.getItem(STORAGE_KEY);
  if (v === "accepted" || v === "declined") return v;
  return "pending";
}

const consent = ref<AdConsent>(readInitial());
const adsenseReady = ref(false);

function markReady() {
  adsenseReady.value = true;
}

function loadAdSenseScript() {
  if (typeof document === "undefined") return;
  const existing = document.querySelector<HTMLScriptElement>(
    `script[${SCRIPT_ATTR}]`,
  );
  if (existing) {
    if (typeof window !== "undefined" && window.adsbygoogle) {
      markReady();
    } else {
      existing.addEventListener("load", markReady, { once: true });
    }
    return;
  }
  const s = document.createElement("script");
  s.async = true;
  s.src = `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT_ID}`;
  s.setAttribute(SCRIPT_ATTR, "1");
  s.addEventListener("load", markReady, { once: true });
  document.head.appendChild(s);
}

if (typeof window !== "undefined" && consent.value === "accepted") {
  loadAdSenseScript();
}

function accept() {
  consent.value = "accepted";
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(STORAGE_KEY, "accepted");
  }
  loadAdSenseScript();
}

function decline() {
  consent.value = "declined";
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(STORAGE_KEY, "declined");
  }
}

export function useConsent() {
  return { consent, accept, decline, adsenseReady };
}
