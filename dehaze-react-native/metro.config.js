const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const {
  withStorybook,
} = require('@storybook/react-native/metro/withStorybook');
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
    },
  },
  watchFolders: [
    path.resolve(__dirname, 'src'),
    path.resolve(__dirname, '../dehaze-tool/dehaze-sdk-js'),
  ],
};

const finalConfig = mergeConfig(getDefaultConfig(__dirname), config);

module.exports = withStorybook(finalConfig, {
  // When false, removes Storybook from bundle (useful for production)
  enabled: process.env.STORYBOOK_ENABLED === 'true',
  // Path to your storybook config (default: './.rnstorybook')
  configPath: path.resolve(__dirname, '.rnstorybook'),
  // Optional websockets configuration for syncing between devices
  // websockets: {
  //   port: 7007,
  //   host: 'localhost',
  // },
});
