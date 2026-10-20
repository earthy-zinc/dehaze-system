/// <reference types="vite/client" />

/** vite define 注入的应用信息（见 vite.config.ts __APP_INFO__） */
declare const __APP_INFO__: {
  pkg: { name: string; version: string };
  buildTimestamp: number;
};
