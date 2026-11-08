// lint 计划表：决定「哪个包、哪类文件、跑哪个工具」。
// lint-staged.config.mjs 与本目录的 run-lint.mjs 共用本表，避免两处映射漂移。
export const FRONTEND = [
  {
    dir: 'clarity-front-vue',
    eslint: ['ts', 'vue'],
    stylelint: ['scss', 'css', 'vue'],
    prettier: ['js', 'cjs', 'ts', 'json', 'tsx', 'css', 'less', 'scss', 'vue', 'html', 'md'],
  },
  {
    dir: 'clarity-front-react',
    eslint: ['ts', 'tsx', 'js', 'jsx'],
    stylelint: ['less', 'scss', 'css'],
    prettier: ['js', 'cjs', 'json', 'jsx', 'ts', 'tsx', 'css', 'less', 'scss', 'html', 'md'],
  },
  {
    dir: 'clarity-taro',
    eslint: ['ts', 'tsx', 'js', 'jsx'],
    stylelint: ['css', 'less'],
    prettier: ['js', 'cjs', 'json', 'jsx', 'ts', 'tsx', 'css', 'less', 'html', 'md'],
    // rn/ 是独立的 React Native 壳工程，自带 .eslintrc 与依赖（@react-native/eslint-config），
    // 主项目没有它需要的插件，纳入会直接报错；它按自己的工程约定维护。
    exclude: ['rn/'],
  },
  {
    dir: 'clarity-uniapp',
    eslint: ['ts', 'vue'],
    stylelint: ['css', 'less', 'scss'],
    prettier: ['js', 'cjs', 'json', 'jsx', 'ts', 'tsx', 'css', 'less', 'scss', 'vue', 'html', 'md'],
  },
  {
    dir: 'clarity-react-native',
    eslint: ['ts', 'tsx', 'js', 'jsx'],
    prettier: ['js', 'cjs', 'json', 'jsx', 'ts', 'tsx', 'css', 'less', 'scss', 'html', 'md'],
  },
  {
    dir: 'clarity-doc',
    prettier: ['md'],
  },
  {
    dir: 'clarity-sdk-js',
    eslint: ['ts'],
    prettier: ['ts', 'json', 'cjs'],
  },
]

export const extOf = (file) => {
  const i = file.lastIndexOf('.')
  return i === -1 ? '' : file.slice(i + 1).toLowerCase()
}

// 按包分组，返回 [{ cwd, steps: [{ tool, files }] }]
// files 为绝对路径；steps 顺序即执行顺序（eslint -> stylelint -> prettier）。
export const planFor = (files, root) => {
  const plans = []

  for (const pkg of FRONTEND) {
    const base = root + '/' + pkg.dir + '/'
    const owned = files.filter((f) => {
      if (!f.startsWith(base)) return false
      const rel = f.slice(base.length)
      return !(pkg.exclude || []).some((prefix) => rel.startsWith(prefix))
    })
    if (owned.length === 0) continue

    const steps = []
    for (const [tool, exts] of [
      ['eslint', pkg.eslint],
      ['stylelint', pkg.stylelint],
      ['prettier', pkg.prettier],
    ]) {
      if (!exts) continue
      const matched = owned.filter((f) => exts.includes(extOf(f)))
      if (matched.length > 0) steps.push({ tool, files: matched })
    }

    if (steps.length > 0) plans.push({ cwd: pkg.dir, steps })
  }

  return plans
}
