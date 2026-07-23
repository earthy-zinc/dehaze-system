import uni from "@dcloudio/vite-plugin-uni";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [uni()],
  server: {
    open: false,
    port: 5176,
    // H5 开发代理：将 /api 请求转发到后端
    // 注意：algorithm-select 必须在 /api 之前，确保 Python 端点优先匹配
    proxy: {
      "/api/v1/algorithm-select": {
        target: "http://127.0.0.1:8991",
        changeOrigin: true,
      },
      "/api": {
        target: "http://127.0.0.1:8989",
        changeOrigin: true,
        // WebSocket 代理（处理进度推送）
        ws: true,
      },
      "/dataset": {
        target: "http://127.0.0.1:9000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/dataset/, ""),
      },
    },
  },
  css: {
    preprocessorOptions: {
      scss: {
        // 取消sass废弃API的报警
        silenceDeprecations: ["legacy-js-api", "color-functions", "import"],
      },
    },
  },
});
