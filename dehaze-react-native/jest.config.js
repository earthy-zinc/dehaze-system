module.exports = {
  preset: 'react-native',
  setupFiles: ['<rootDir>/jest.setup.js'],
  // node_modules/dehaze-sdk-js 是 08-24 的遗留副本（缺 ai-image 等新面），属死物，
  // 勿依赖、勿直接 rm（会让 jest/metro 丢失解析目标）。三路解析均指向工作区 SDK：
  // 编译 tsconfig paths → ../dehaze-sdk-js/dist/index.d.ts
  // 打包 metro watchFolders → ../dehaze-sdk-js
  // 单测 即下方映射 → ../dehaze-sdk-js/dist/index.cjs（SDK exports 的 require 条件）
  // 任何绕开这三路的解析点（独立脚本、CI 里的 node -e require、新工具链）都会静默
  // 吃到旧副本，接入前务必先确认解析目标
  moduleNameMapper: {
    '^dehaze-sdk-js$': '<rootDir>/../dehaze-sdk-js/dist/index.cjs',
  },
  // RN preset 默认不转译 node_modules，而这些依赖随包发布 ESM，
  // 必须显式放行，否则 jest 以 CJS 解析时报 "Unexpected token 'export'"
  transformIgnorePatterns: [
    'node_modules/(?!(jest-)?@?react-native|@react-native-community|@react-navigation|react-native-.*)',
  ],
};
