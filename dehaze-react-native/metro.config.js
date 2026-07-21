const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const path = require('path');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */

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
      // 让 dehaze-sdk-js（位于 watchFolders 但无自身 node_modules）能解析到 RN 项目的依赖
      '@babel/runtime': path.resolve(__dirname, 'node_modules/@babel/runtime'),
    },
    // SDK 位于 watchFolders，默认只会在 SDK 自身和其父级查找 node_modules；
    // 显式追加 RN 项目的 node_modules，确保 @babel/runtime 等可被解析
    nodeModulesPaths: [
      path.resolve(__dirname, 'node_modules'),
    ],
  },
  watchFolders: [
    path.resolve(__dirname, 'src'),
    path.resolve(__dirname, '../dehaze-tool/dehaze-sdk-js'),
  ],
};

const finalConfig = mergeConfig(getDefaultConfig(__dirname), config);

// Storybook 仅在显式启用时加载，避免其自定义 resolver 破坏常规模块解析
if (process.env.STORYBOOK_ENABLED === 'true') {
  const { withStorybook } = require('@storybook/react-native/metro/withStorybook');
  module.exports = withStorybook(finalConfig, {
    enabled: true,
    configPath: path.resolve(__dirname, '.rnstorybook'),
  });
} else {
  module.exports = finalConfig;
}
