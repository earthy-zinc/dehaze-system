module.exports = {
  root: true,
  extends: ['@react-native', 'plugin:storybook/recommended'],
  parserOptions: {
    requireConfigFile: false,
  },
  overrides: [
    {
      files: ['*.js'],
      excludedFiles: ['metro.config.cjs', '.eslintrc.js'],
    },
    {
      files: ['metro.config.js'],
      parserOptions: {
        requireConfigFile: false,
      },
    },
  ],
};
