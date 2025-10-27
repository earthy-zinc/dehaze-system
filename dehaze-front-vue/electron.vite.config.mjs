// electron.vite.config.mjs
import { defineConfig, defineViteConfig, mergeConfig } from "electron-vite";
import { resolve } from "path";
import viteConfig from "./vite.config.ts";

export default defineConfig({
  main: {
    build: {
      lib: {
        entry: resolve(__dirname, "electron/main/index.ts"),
      },
    },
  },
  preload: {
    build: {
      lib: {
        entry: resolve(__dirname, "electron/preload/index.ts"),
      },
    },
  },
  renderer: defineViteConfig((config) => {
    return mergeConfig(viteConfig(config), {
      root: resolve(__dirname),
      resolve: {
        alias: {
          "@": resolve(__dirname, "src"),
        },
      },
      server: {
        host: "0.0.0.0",
        port: 3000,
      },
      build: {
        rollupOptions: {
          input: resolve(__dirname, "index.html"),
        },
      },
    });
  }),
});
