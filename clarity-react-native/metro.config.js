const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const path = require('path');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */

// 源码直连：node_modules/clarity-sdk-js 是 file: 安装出来的裁剪副本（只有 dist + src/types），
// 改源码不会同步；这里把包名直接指向工作区源码入口，配合 watchFolders 获得热更新。
const sdkEntry = path.resolve(__dirname, '../clarity-sdk-js/index.ts');

const config = {
  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: true,
      },
    }),
  },
  resolver: {
    extraNodeModules: {
      '@': path.resolve(__dirname, 'src'),
      // 让 clarity-sdk-js（位于 watchFolders 但无自身 node_modules）能解析到 RN 项目的依赖
      '@babel/runtime': path.resolve(__dirname, 'node_modules/@babel/runtime'),
    },
    // SDK 位于 watchFolders，默认只会在 SDK 自身和其父级查找 node_modules；
    // 显式追加 RN 项目的 node_modules，确保 @babel/runtime 等可被解析
    nodeModulesPaths: [path.resolve(__dirname, 'node_modules')],
    resolveRequest: (context, moduleName, platform) => {
      if (moduleName === 'clarity-sdk-js') {
        return { type: 'sourceFile', filePath: sdkEntry };
      }
      return context.resolveRequest(context, moduleName, platform);
    },
  },
  watchFolders: [
    path.resolve(__dirname, 'src'),
    path.resolve(__dirname, '../clarity-sdk-js'),
  ],
};

const finalConfig = mergeConfig(getDefaultConfig(__dirname), config);

// Storybook 仅在显式启用时加载，避免其自定义 resolver 破坏常规模块解析
if (process.env.STORYBOOK_ENABLED === 'true') {
  const {
    withStorybook,
  } = require('@storybook/react-native/metro/withStorybook');
  module.exports = withStorybook(finalConfig, {
    enabled: true,
    configPath: path.resolve(__dirname, '.rnstorybook'),
  });
} else {
  module.exports = finalConfig;
}
