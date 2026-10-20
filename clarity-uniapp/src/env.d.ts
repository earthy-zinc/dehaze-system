/// <reference types="vite/client" />

/** vite define 注入的应用信息（见 vite.config.ts __APP_INFO__） */
declare const __APP_INFO__: {
  pkg: { version: string };
};

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<
    Record<string, never>,
    Record<string, never>,
    any
  >;
  export default component;
}
