module.exports = {
  root: true,
  extends: ['@react-native', 'plugin:storybook/recommended'],
  parserOptions: {
    requireConfigFile: false,
  },
  rules: {
    '@typescript-eslint/no-unused-vars': [
      'error',
      { varsIgnorePattern: '^_', argsIgnorePattern: '^_' },
    ],
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
