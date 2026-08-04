import path from "node:path";
import { fileURLToPath } from "node:url";
import uni from "@dcloudio/vite-plugin-uni";
import { defineConfig, loadEnv } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  // 与 src/api/constants.ts 共用同一环境变量（VITE_API_HOST）
  const apiHost = env.VITE_API_HOST || "http://127.0.0.1:8989";

  return {
    plugins: [uni()],
    server: {
      open: false,
      port: 5176,
      // H5 开发代理：将 /api 请求转发到后端
      proxy: {
        "/api": {
          target: apiHost,
          changeOrigin: true,
          // WebSocket 代理（处理进度推送）
          ws: true,
        },
      },
    },
    css: {
      preprocessorOptions: {
        scss: {
          // 取消sass废弃API的报警
          silenceDeprecations: ["legacy-js-api", "color-functions", "import"],
          // 全局注入设计令牌，使任意 <style lang="scss"> 块可直接使用 $color-* / $spacing-* 等变量
          additionalData: `@import "${path.resolve(__dirname, "src/styles/variables.scss").replace(/\\/g, "/")}";`,
        },
      },
    },
  };
});
