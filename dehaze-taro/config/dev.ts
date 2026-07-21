import type {UserConfigExport} from "@tarojs/cli";

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
        // H5 开发环境经 devServer 代理转发，规避浏览器 CORS 限制
        // 小程序端不受影响（小程序不存在跨域问题，直连绝对地址）
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
