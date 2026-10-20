/**
 * Jest 全局 mock：这些库依赖原生模块，单测环境无原生运行时，
 * 用官方提供的 jest 替身替换，避免 require 时抛错。
 */
require('react-native-gesture-handler/jestSetup');

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

jest.mock('react-native-reanimated', () =>
  require('react-native-reanimated/mock'),
);
