import vue from "@vitejs/plugin-vue";
import vueJsx from "@vitejs/plugin-vue-jsx";
import { resolve } from "path";

import UnoCSS from "unocss/vite";
import AutoImport from "unplugin-auto-import/vite";
import IconsResolver from "unplugin-icons/resolver";
import Icons from "unplugin-icons/vite";
import { ElementPlusResolver } from "unplugin-vue-components/resolvers";
import Components from "unplugin-vue-components/vite";
import {
  ConfigEnv,
  defineConfig,
  loadEnv,
  PluginOption,
  UserConfig,
} from "vite";
import { mockDevServerPlugin } from "vite-plugin-mock-dev-server";

import { createSvgIconsPlugin } from "vite-plugin-svg-icons";
import {
  dependencies,
  devDependencies,
  engines,
  name,
  version,
} from "./package.json";

// https://devtools-next.vuejs.org/

/** 平台的名称、版本、运行所需的`node`版本、依赖、构建时间的类型提示 */
const __APP_INFO__ = {
  pkg: { name, version, engines, dependencies, devDependencies },
  buildTimestamp: Date.now(),
};

const pathSrc = resolve(__dirname, "src");
//  https://cn.vitejs.dev/config
export default defineConfig(({ command, mode }: ConfigEnv): UserConfig => {
  const env = loadEnv(mode, process.cwd());
  // 样张静态基址随环境而异，由部署侧注入：生产构建缺失即失败，避免打包出指向不存在域名的页面
  if (command === "build" && !env.VITE_DATASET_HOST) {
    throw new Error(
      "缺少 VITE_DATASET_HOST：请在构建环境注入数据集静态服务地址（如 VITE_DATASET_HOST=https://cdn.<域名>:19000），否则样张图片不可用"
    );
  }
  return {
    resolve: {
      alias: {
        "@": pathSrc,
      },
    },
    css: {
      // CSS 预处理器
      preprocessorOptions: {
        // 定义全局 SCSS 变量
        scss: {
          additionalData: `
            @use "@/styles/variables.scss" as *;
          `,
        },
      },
    },
    server: {
      // 允许IP访问
      host: "0.0.0.0",
      // 应用端口 (默认:5174)
      port: Number(env.VITE_APP_PORT),
      // 运行是否自动打开浏览器
      open: false,
      watch: {
        /**
         * 链接的工作区包 clarity-sdk-js 通过 dist 产物被引入（node_modules 软链），
         * 其重建会触发 vite 整页 reload（日志形如 `page reload .../clarity-sdk-js/dist/index.js`），
         * 打断正在做页面评审/测试的同学（评估期间曾多次整页刷新）。
         * 该目录是构建产物而非应用源码，dev 下不监听：SDK 改动后手动刷新页面即可。
         * 如需恢复监听，删除本 watch 配置项即可。
         */
        ignored: ["**/clarity-sdk-js/**"],
      },
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
     * 预览服务：页面级功能测试走构建产物，避开 dev server 的整页 reload——SDK（软链引入）
     * 的 dist 重建会让 optimizeDeps 重新预构建并全量刷新，`server.watch.ignored` 只作用于
     * 文件监听、覆盖不到该路径。代理口径与 dev 保持一致，便于同一套页面用例切换。
     *
     * 端口与 dev 错开（VITE_PREVIEW_PORT），两者可并存：测试期间无需停掉别人的 dev server。
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
    /**
     * unocss 与 unplugin 系插件的类型来自各自依赖树里的旧 vite 副本，与本项目 vite 的结构化
     * Plugin 类型不能互相赋值；以下 4 处仅做类型对齐，插件实例与运行行为不受影响
     *（依赖去重属独立任务）。
     */
    plugins: [
      vue(),
      // jsx、tsx语法支持
      vueJsx(),
      // MOCK 服务
      env.VITE_MOCK_DEV_SERVER === "true" ? mockDevServerPlugin() : null,
      UnoCSS({
        hmrTopLevelAwait: false,
      }) as PluginOption,
      // 自动导入参考： https://github.com/sxzz/element-plus-best-practices/blob/main/vite.config.ts
      AutoImport({
        // 自动导入 Vue 相关函数，如：ref, reactive, toRef 等
        imports: ["vue", "@vueuse/core", "pinia", "vue-router", "vue-i18n"],
        resolvers: [
          // 自动导入 Element Plus 相关函数，如：ElMessage, ElMessageBox... (带样式)
          ElementPlusResolver(),
          // 自动导入图标组件
          IconsResolver({}),
        ],
        eslintrc: {
          // 是否自动生成 eslint 规则，建议生成之后设置 false
          enabled: false,
          // 指定自动导入函数 eslint 规则的文件
          filepath: "./.eslintrc-auto-import.json",
          globalsPropValue: true,
        },
        // 是否在 vue 模板中自动导入
        vueTemplate: true,
        // 指定自动导入函数TS类型声明文件路径 (false:关闭自动生成)
        dts: false,
        // dts: "src/typings/auto-imports.d.ts",
      }) as PluginOption,
      Components({
        resolvers: [
          // 自动导入 Element Plus 组件
          ElementPlusResolver(),
          // 自动注册图标组件
          IconsResolver({
            // element-plus图标库，其他图标库 https://icon-sets.iconify.design/
            enabledCollections: ["ep"],
          }),
        ],
        // 指定自定义组件位置(默认:src/components)
        dirs: ["src/components", "src/**/components"],
        // 指定自动导入组件TS类型声明文件路径 (false:关闭自动生成)
        dts: false,
        // dts: "src/typings/components.d.ts",
      }) as PluginOption,
      Icons({
        // 自动安装图标库
        autoInstall: true,
      }) as PluginOption,
      createSvgIconsPlugin({
        // 指定需要缓存的图标文件夹
        iconDirs: [resolve(pathSrc, "assets/icons")],
        // 指定symbolId格式
        symbolId: "icon-[dir]-[name]",
      }),
      // VueDevTools({
      //   openInEditorHost: `http://localhost:${env.VITE_APP_PORT}`,
      // }),
    ],
    // 预加载项目必需的组件
    optimizeDeps: {
      include: [
        "vue",
        "vue-router",
        "pinia",
        "axios",
        "@vueuse/core",
        "sortablejs",
        "path-to-regexp",
        "echarts",
        "@wangeditor/editor",
        "@wangeditor/editor-for-vue",
        "vue-i18n",
        "path-browserify",
        "element-plus/es/components/form/style/css",
        "element-plus/es/components/form-item/style/css",
        "element-plus/es/components/button/style/css",
        "element-plus/es/components/input/style/css",
        "element-plus/es/components/input-number/style/css",
        "element-plus/es/components/switch/style/css",
        "element-plus/es/components/upload/style/css",
        "element-plus/es/components/menu/style/css",
        "element-plus/es/components/col/style/css",
        "element-plus/es/components/icon/style/css",
        "element-plus/es/components/row/style/css",
        "element-plus/es/components/tag/style/css",
        "element-plus/es/components/dialog/style/css",
        "element-plus/es/components/loading/style/css",
        "element-plus/es/components/radio/style/css",
        "element-plus/es/components/radio-group/style/css",
        "element-plus/es/components/popover/style/css",
        "element-plus/es/components/scrollbar/style/css",
        "element-plus/es/components/tooltip/style/css",
        "element-plus/es/components/dropdown/style/css",
        "element-plus/es/components/dropdown-menu/style/css",
        "element-plus/es/components/dropdown-item/style/css",
        "element-plus/es/components/sub-menu/style/css",
        "element-plus/es/components/menu-item/style/css",
        "element-plus/es/components/divider/style/css",
        "element-plus/es/components/card/style/css",
        "element-plus/es/components/link/style/css",
        "element-plus/es/components/breadcrumb/style/css",
        "element-plus/es/components/breadcrumb-item/style/css",
        "element-plus/es/components/table/style/css",
        "element-plus/es/components/tree-select/style/css",
        "element-plus/es/components/table-column/style/css",
        "element-plus/es/components/select/style/css",
        "element-plus/es/components/option/style/css",
        "element-plus/es/components/pagination/style/css",
        "element-plus/es/components/tree/style/css",
        "element-plus/es/components/alert/style/css",
        "element-plus/es/components/radio-button/style/css",
        "element-plus/es/components/checkbox-group/style/css",
        "element-plus/es/components/checkbox/style/css",
        "element-plus/es/components/tabs/style/css",
        "element-plus/es/components/tab-pane/style/css",
        "element-plus/es/components/rate/style/css",
        "element-plus/es/components/date-picker/style/css",
        "element-plus/es/components/notification/style/css",
        "element-plus/es/components/image/style/css",
        "element-plus/es/components/statistic/style/css",
        "element-plus/es/components/watermark/style/css",
        "element-plus/es/components/config-provider/style/css",
        "element-plus/es/components/text/style/css",
        "element-plus/es/components/drawer/style/css",
        "element-plus/es/components/color-picker/style/css",
      ],
      exclude: [],
    },
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
          // manualChunks: {
          //   "vue-i18n": ["vue-i18n"],
          // },
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
