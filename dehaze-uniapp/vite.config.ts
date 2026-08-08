import path from "node:path";
import { fileURLToPath } from "node:url";
import uni from "@dcloudio/vite-plugin-uni";
import { createSvgIconsPlugin } from "vite-plugin-svg-icons";
import { defineConfig, loadEnv } from "vite";
// 构建时注入应用版本号（供前端日志 app_version 字段）
import { version as APP_VERSION } from "./package.json";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pathSrc = path.resolve(__dirname, "src");

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  // 与 src/api/constants.ts 共用同一环境变量（VITE_API_HOST）
  const apiHost = env.VITE_API_HOST || "http://127.0.0.1:8989";

  return {
    plugins: [
      uni(),
      createSvgIconsPlugin({
        // 需要缓存的图标文件夹（svg sprite 自动注册）
        iconDirs: [path.resolve(pathSrc, "assets/icons")],
        // symbolId 格式：SvgIcon 组件内拼接 #icon-${name}
        symbolId: "icon-[name]",
      }),
    ],
    // 注入应用信息：供前端日志 app_version / env 使用（与 dehaze-front-react 的 __APP_INFO__ 一致）
    define: {
      __APP_INFO__: JSON.stringify({
        pkg: { version: APP_VERSION },
      }),
    },
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
