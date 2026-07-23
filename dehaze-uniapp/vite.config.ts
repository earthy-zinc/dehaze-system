import path from "node:path";
import { fileURLToPath } from "node:url";
import uni from "@dcloudio/vite-plugin-uni";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

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
        rewrite: (p) => p.replace(/^\/dataset/, ""),
      },
    },
  },
  css: {
    preprocessorOptions: {
      scss: {
        // 取消sass废弃API的报警
        silenceDeprecations: ["legacy-js-api", "color-functions", "import"],
        // 全局注入设计令牌，使任意 <style lang="scss"> 块可直接使用 $color-* / $spacing-* 等变量
        additionalData: `@import "${path.resolve(__dirname, "src/styles/variables.scss")}";`,
      },
    },
  },
});
