/// <reference types="vite/client" />

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
