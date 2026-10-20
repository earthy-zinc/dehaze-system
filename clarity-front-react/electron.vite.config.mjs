// electron.vite.config.mjs
import {
  defineConfig,
  defineViteConfig,
  mergeConfig,
  loadEnv,
} from "electron-vite";
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
    const { mode } = config;
    const env = loadEnv(mode, process.cwd());

    return mergeConfig(viteConfig(config), {
      root: resolve(__dirname),
      server: {
        port: env.VITE_ELECTRON_PORT,
      },
      build: {
        rollupOptions: {
          input: resolve(__dirname, "index.html"),
        },
      },
    });
  }),
});
