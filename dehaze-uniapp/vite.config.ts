import uni from "@dcloudio/vite-plugin-uni";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [uni()],
  server: {
    open: false,
    port: 5176,
    // H5 开发代理：将 /api 请求转发到后端
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8989",
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
      },
    },
  },
});
