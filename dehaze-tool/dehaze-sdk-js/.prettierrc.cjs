module.exports = {
  // (x)=>{},单个参数箭头函数是否显示小括号。(always:始终显示;avoid:省略括号。默认:always)
  arrowParens: "always",
  // 对象字面量的括号之间打印空格 (true - Example: { foo: bar } ; false - Example: {foo:bar})
  bracketSpacing: true,
  // 指定 HTML 文件的空格敏感度 (css|strict|ignore;默认css)
  htmlWhitespaceSensitivity: "css",
  // 在 JSX 中使用单引号替代双引号，默认false
  jsxSingleQuote: false,
  // 每行最多字符数量，超出换行(默认80)
  printWidth: 100,
  // 超出打印宽度 (always | never | preserve )
  proseWrap: "preserve",
  // 对象属性是否使用引号(as-needed | consistent | preserve;默认as-needed)
  quoteProps: "as-needed",
  // 结尾添加分号
  semi: true,
  // 使用单引号 (true:单引号;false:双引号)
  singleQuote: false,
  // 缩进空格数，默认2个空格
  tabWidth: 2,
  // 元素末尾是否加逗号，默认es5
  trailingComma: "es5",
  // 指定缩进方式，空格或tab，默认false，即使用空格
  useTabs: false,
  // 换行符
  endOfLine: "auto",
};
