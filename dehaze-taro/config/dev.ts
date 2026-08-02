import type { UserConfigExport } from "@tarojs/cli";

export default {
  logger: {
    quiet: false,
    stats: true,
  },
  mini: {
    webpackChain: (chain, webpack) => {
      chain.merge({
        plugin: {
          install: {
            plugin: require("terser-webpack-plugin"),
            args: [
              {
                terserOptions: {
                  compress: true, // 默认使用terser压缩
                  // mangle: false,
                  keep_classnames: true, // 不改变class名称
                  keep_fnames: true, // 不改变函数名称
                },
              },
            ],
          },
        },
      });
    },
  },
  h5: {
    devServer: {
      open: false,
      port: 5175,
      proxy: {
        "/api": {
          target: "http://localhost:8989",
          changeOrigin: true,
          // 浏览器同源请求会带上 Origin 头，服务端转发时移除，
          // 避免被后端 CORS 白名单拦截（403）。代理转发属同源请求，无需 CORS。
          onProxyReq: (proxyReq: { removeHeader: (name: string) => void }) => {
            proxyReq.removeHeader("origin");
          },
        },
      },
    },
  },
} satisfies UserConfigExport<"webpack5">;
