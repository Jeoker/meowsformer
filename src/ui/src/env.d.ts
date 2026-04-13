/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ADSENSE_SLOT_RESULT?: string;
  readonly VITE_ADSENSE_SLOT_ABOUT_MID?: string;
  readonly VITE_ADSENSE_SLOT_ABOUT_BOTTOM?: string;
}

declare global {
  interface Window {
    adsbygoogle?: unknown[];
  }
}

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<{}, {}, any>;
  export default component;
}

export {};
