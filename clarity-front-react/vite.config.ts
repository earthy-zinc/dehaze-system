import react from "@vitejs/plugin-react";
import { resolve } from "path";
import UnoCSS from "unocss/vite";

import { ConfigEnv, defineConfig, loadEnv, UserConfig } from "vite";
import { mockDevServerPlugin } from "vite-plugin-mock-dev-server";
import svgr from "vite-plugin-svgr";
import {
  dependencies,
  devDependencies,
  engines,
  name,
  version,
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
      // 应用端口 (默认:5173)
      port: Number(env.VITE_APP_PORT),
      // 运行是否自动打开浏览器
      open: false,
      proxy: {
        /** 代理前缀为 /api 的请求  */
        [env.VITE_BASE_API]: {
          changeOrigin: true,
          // 接口地址
          target: env.VITE_API_URL,
        },
        /** 流式 ASR WebSocket：后端返回相对路径，浏览器按页面 origin 同源连接，由代理转发（需显式开启 ws） */
        "/ws": {
          target: env.VITE_API_URL,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    /**
     * 预览服务：页面级功能测试走构建产物（见 .env.test 与 e2e/），避开 dev server 在源码或
     * SDK 产物变动时的整页 reload。端口与 dev 错开（VITE_PREVIEW_PORT），两者可并存。
     * 以 `vite preview --mode test` 启动：代理目标与端口定义在 .env.test（生产由 nginx
     * 同源反代 /api、没有 VITE_API_URL，按默认 production 模式读 config 会拿到空代理目标）。
     */
    preview: {
      host: "0.0.0.0",
      port: Number(env.VITE_PREVIEW_PORT),
      proxy: {
        [env.VITE_BASE_API]: {
          changeOrigin: true,
          target: env.VITE_API_URL,
        },
        "/ws": {
          target: env.VITE_API_URL,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    plugins: [
      react(),
      // MOCK 服务
      env.VITE_MOCK_DEV_SERVER === "true" ? mockDevServerPlugin() : null,
      UnoCSS({
        hmrTopLevelAwait: false,
      }) as any,
      svgr({
        svgrOptions: {
          icon: true, // 将 SVG 转换为 React 组件时自动调整为图标尺寸
          svgProps: { focusable: "false" }, // 为 SVG 添加默认属性
        },
        include: "src/assets/icons/*.svg",
      }),
    ],
    // 构建配置
    build: {
      chunkSizeWarningLimit: 2000, // 消除打包大小超过500kb警告
      minify: "terser", // Vite 2.6.x 以上需要配置 minify: "terser", terserOptions 才能生效
      terserOptions: {
        compress: {
          keep_infinity: true, // 防止 Infinity 被压缩成 1/0，这可能会导致 Chrome 上的性能问题
          // 生产构建去除 console/debugger；mode=test（页面测试基座，.env.test）保留它们，
          // 页面级断言依赖应用自身打印的控制台报错
          drop_console: mode !== "test",
          drop_debugger: mode !== "test",
        },
        format: {
          comments: false, // 删除注释
        },
      },
      rollupOptions: {
        output: {
          // 用于从入口点创建的块的打包输出格式[name]表示文件名,[hash]表示该文件内容hash值
          entryFileNames: "js/[name].[hash].js",
          // 用于命名代码拆分时创建的共享块的输出命名
          chunkFileNames: "js/[name].[hash].js",
          // 用于输出静态资源的命名，[ext]表示文件扩展名
          assetFileNames: (assetInfo: any) => {
            const info = assetInfo.name.split(".");
            let extType = info[info.length - 1];
            // console.log('文件信息', assetInfo.name)
            if (
              /\.(mp4|webm|ogg|mp3|wav|flac|aac)(\?.*)?$/i.test(assetInfo.name)
            ) {
              extType = "media";
            } else if (/\.(png|jpe?g|gif|svg)(\?.*)?$/.test(assetInfo.name)) {
              extType = "img";
            } else if (/\.(woff2?|eot|ttf|otf)(\?.*)?$/i.test(assetInfo.name)) {
              extType = "fonts";
            }
            return `${extType}/[name].[hash].[ext]`;
          },
        },
      },
    },
    define: {
      __APP_INFO__: JSON.stringify(__APP_INFO__),
    },
  };
});
