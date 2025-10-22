module.exports = {
  root: true,
  extends: '@react-native',
  parserOptions: {
    requireConfigFile: false,
  },
  overrides: [
    {
      files: ['*.js'],
      excludedFiles: ['metro.config.js', '.eslintrc.js'],
    },
    {
      files: ['metro.config.js'],
      parserOptions: {
        requireConfigFile: false,
      },
    },
  ],
};