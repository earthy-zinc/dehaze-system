import type {UserConfigExport} from "@tarojs/cli";

export default {
  mini: {},
  h5: {
    /**
     * WebpackChain 插件配置
     * @docs https://github.com/neutrinojs/webpack-chain
     */
    webpackChain(chain) {
      /**
       * 如果 h5 端编译后体积过大，可以使用 webpack-bundle-analyzer 插件对打包体积进行分析。
       * @docs https://github.com/webpack-contrib/webpack-bundle-analyzer
       */
      chain
        .plugin("analyzer")
        .use(require("webpack-bundle-analyzer").BundleAnalyzerPlugin, []);

      chain.plugin("compression").use(require("compression-webpack-plugin"), [
        {
          algorithm: "gzip",
          test: /\.(js|css|html|svg)$/,
          threshold: 8192,
          minRatio: 0.8,
        },
      ]);

      chain.optimization.splitChunks({
        chunks: "all",
        maxInitialRequests: 5,
        maxAsyncRequests: 5,
        cacheGroups: {
          vendor: {
            name: "vendor",
            priority: 10,
            test: /[\\/]node_modules[\\/]/,
            chunks: "initial",
            maxSize: 244 * 1024,
            maxInitialRequests: 5, // 限制初始请求数
            minSize: 20 * 1024, // 设置最小块大小
          },
          react: {
            name: "react",
            test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
            priority: 20,
            chunks: "all",
          },
        },
      });
    },
  },
} satisfies UserConfigExport<"webpack5">;
