import { resolve } from "path";
import react from "@vitejs/plugin-react";
import { defineConfig, UserConfig, ConfigEnv, loadEnv } from "vite";
import {
  name,
  version,
  engines,
  dependencies,
  devDependencies,
} from "./package.json";

/** 平台的名称、版本、运行所需的`node`版本、依赖、构建时间的类型提示 */
const __APP_INFO__ = {
  pkg: { name, version, engines, dependencies, devDependencies },
  buildTimestamp: Date.now(),
};

const pathSrc = resolve(__dirname, "src");
// https://vitejs.dev/config/
export default defineConfig(({ mode }: ConfigEnv): UserConfig => {
  const env = loadEnv(mode, process.cwd());
  return {
    resolve: {
      alias: {
        "@": pathSrc,
      },
    },

    server: {
      // 允许IP访问
      host: "0.0.0.0",
      // 应用端口 (默认:3000)
      port: Number(env.VITE_APP_PORT),
      // 运行是否自动打开浏览器
      open: true,
      proxy: {
        /** 代理前缀为 /dev-api 的请求  */
        [env.VITE_APP_BASE_API]: {
          changeOrigin: true,
          // 接口地址
          target: env.VITE_APP_API_URL,
          rewrite: (path) =>
            path.replace(new RegExp("^" + env.VITE_APP_BASE_API), ""),
        },
      },
    },
    plugins: [react()],
    define: {
      __APP_INFO__: JSON.stringify(__APP_INFO__),
    },
  };
});
