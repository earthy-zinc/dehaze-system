import { defineConfig, type UserConfigExport } from "@tarojs/cli";
import path from "path";
import TsconfigPathsPlugin from "tsconfig-paths-webpack-plugin";
import devConfig from "./dev";
import prodConfig from "./prod";
// 构建时注入应用版本号（供前端日志 app_version 字段）
import { version as APP_VERSION } from "../package.json";

// https://taro-docs.jd.com/docs/next/config#defineconfig-辅助函数
export default defineConfig<"webpack5">(async (merge, { command, mode }) => {
  const baseConfig: UserConfigExport<"webpack5"> = {
    projectName: "dehaze-taro",
    date: "2025-9-22",
    designWidth: 750,
    // 官方默认换算比例：750 设计稿下 1rpx = 屏宽/750。
    // 此前误配为 2 倍（750:2），导致 H5 端 rpx→rem 换算放大一倍、页面整体 2 倍变大，
    // 且 html 根字号被钳制在 minRootSize=20px 后 1rpx 直接等于 1px。
    deviceRatio: {
      640: 2.34 / 2,
      750: 1,
      375: 2,
      828: 1.81 / 2,
    },
    alias: {
      "@": path.resolve(__dirname, "..", "src"),
    },
    sourceRoot: "src",
    outputRoot: `dist/${process.env.TARO_ENV}`,
    plugins: ["@tarojs/plugin-http", "@tarojs/plugin-platform-harmony-cpp"],
    // 显式注入 process.env.* 变量，确保 webpack DefinePlugin 在编译期完成静态替换
    // Taro 内置仅注入 TARO_ENV；TARO_APP_* 需 .env 文件定义才会注入，本项目无 .env，
    // 若不显式声明，代码中保留 process.env.* 原文，浏览器运行时报 ReferenceError: process is not defined
    defineConstants: {
      "process.env.TARO_ENV": JSON.stringify(process.env.TARO_ENV),
      "process.env.TARO_APP_JAVA_BASE_URL": JSON.stringify(
        process.env.TARO_APP_JAVA_BASE_URL ?? ""
      ),
      "process.env.TARO_APP_VERSION": JSON.stringify(APP_VERSION),
      "process.env.NODE_ENV": JSON.stringify(
        process.env.NODE_ENV ?? "production"
      ),
    },
    copy: {
      patterns: [],
      options: {},
    },
    framework: "react",
    compiler: "webpack5",
    cache: {
      enable: true, // Webpack 持久化缓存配置，建议开启。默认配置请参考：https://docs.taro.zone/docs/config-detail#cache
    },
    module: {
      rules: [
        {
          test: /\.js$/,
          use: "babel-loader",
          exclude: /node_modules/,
        },
      ],
    },
    mini: {
      postcss: {
        pxtransform: {
          enable: true,
          config: {},
        },
        cssModules: {
          enable: false, // 默认为 false，如需使用 css modules 功能，则设为 true
          config: {
            namingPattern: "module", // 转换模式，取值为 global/module
            generateScopedName: "[name]__[local]___[fullhash:base64:5]",
          },
        },
      },
      webpackChain(chain) {
        chain.resolve.plugin("tsconfig-paths").use(TsconfigPathsPlugin);
      },
    },
    h5: {
      publicPath: "/",
      staticDirectory: "static",
      output: {
        filename: "js/[name].[fullhash:8].js",
        chunkFilename: "js/[name].[chunkhash:8].js",
      },
      miniCssExtractPluginOption: {
        ignoreOrder: true,
        filename: "css/[name].[fullhash:8].css",
        chunkFilename: "css/[name].[chunkhash].css",
      },
      postcss: {
        autoprefixer: {
          enable: true,
          config: {},
        },
        cssModules: {
          enable: false, // 默认为 false，如需使用 css modules 功能，则设为 true
          config: {
            namingPattern: "module", // 转换模式，取值为 global/module
            generateScopedName: "[name]__[local]___[hash:base64:5]",
          },
        },
      },
      webpackChain(chain) {
        chain.resolve.plugin("tsconfig-paths").use(TsconfigPathsPlugin);
      },
    },
    rn: {
      appName: "taroDemo",
      postcss: {
        cssModules: {
          enable: false, // 默认为 false，如需使用 css modules 功能，则设为 true
        },
      },
    },
    harmony: {
      // 当前仅支持使用 Vite 编译鸿蒙应用
      // @ts-ignore Taro 4.1.7 类型定义中 IHarmonyConfig 缺少 compiler 属性，运行时需要
      compiler: "vite",
      // Note: 鸿蒙工程路径，可以参考 [鸿蒙应用创建导读](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V2/start-with-ets-stage-0000001477980905-V2) 创建
      projectPath: path.resolve(__dirname, "..", "harmory"),
      // Taro 项目编译到对应鸿蒙模块名，默认为 entry
      hapName: "entry",
    },
  };
  if (process.env.NODE_ENV === "development") {
    // 本地开发构建配置（不混淆压缩）
    return merge({}, baseConfig, devConfig);
  }
  // 生产构建配置（默认开启压缩混淆等）
  return merge({}, baseConfig, prodConfig);
});
