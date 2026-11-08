/**
 * RN 构建专用的样式兼容层，只作用于 `taro build --type rn`，不影响 h5 与小程序。
 *
 * Taro 的 RN 样式链路会把 RN 不认识的写法原样透传给 css-to-react-native，
 * 由后者抛错中断编译。这里在样式转换前补齐 RN 缺失的几项能力：
 *
 * 1. CSS 变量：项目变量集中在 src/app.less 的 :root，页面 less 不 @import，
 *    Taro 的 postcss-css-variables 拿不到定义会写成字符串 undefined，故关掉它改由这里内联。
 * 2. 单位：rn 平台的 pxtransform 只转换 px，rpx 会原样透传导致解析失败，这里按设计稿换算。
 *    vh / vw 不在此列——css-to-react-native 原生支持，会转成 scaleVu2dp 做运行时适配。
 * 3. 无法表达的值：渐变降级为透明；env() / constant() 安全区域、calc() 表达式整条删除。
 *
 * shortcut: em 按 14px 基准字号折算成固定 px，字号层级变化后会偏；
 * 项目里 em 仅 6 处，改为显式 px 后即可删掉这条转换。
 * shortcut: app.less 里查不到的变量会让整条声明被删除，样式表现为「静默不生效」；
 * 排查某处样式丢失时先对变量名，补进 app.less 即可消除，长期应改成显式的主题 token。
 */
const fs = require("node:fs");
const path = require("node:path");

// 用插件自身位置定位，避免依赖 Metro worker 的工作目录
const VARIABLE_FILE = path.resolve(__dirname, "../src/app.less");

let variableCache = null;
let variableMtime = 0;

// app.less 缺失或不可读时直接抛错：静默退化成空变量表会让全站颜色与间距消失，且构建仍然成功
function loadVariables() {
  const stat = fs.statSync(VARIABLE_FILE);
  if (variableCache && stat.mtimeMs === variableMtime) return variableCache;

  variableMtime = stat.mtimeMs;
  variableCache = {};
  const source = fs.readFileSync(VARIABLE_FILE, "utf8");
  // 必须行首锚定：app.less 里有 `// ---- 设计令牌 ----` 这类注释，
  // 不锚定会让 `----` 被当成变量名，并吞掉紧随其后的真实定义
  const re = /^[ \t]*(--[\w-]+)\s*:\s*([^;]+);/gm;
  let match;
  while ((match = re.exec(source))) {
    variableCache[match[1]] = match[2].trim();
  }
  return variableCache;
}

const VAR_REF = /var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)/g;
const GRADIENT = /(linear|radial|conic)-gradient\(/i;
const UNRESOLVABLE = /\b(env|constant|calc)\(/i;
// 前后加边界，避免把 url(icon1rpx.png)、1e2rpx 这类内容改坏
const CONVERTERS = [
  [/(?<![\w.])(\d*\.?\d+)rpx(?![\w.])/gi, (n) => n / 2],
  [/(?<![\w.])(\d*\.?\d+)em(?![\w.])/gi, (n) => n * 14],
];
const DECLARED_PX = /(?<![\w.])(\d*\.?\d+)px(?![\w.])/gi;

function inlineVariables(value) {
  const variables = loadVariables();
  let unresolved = false;
  const next = value.replace(VAR_REF, (_, name, fallback) => {
    if (variables[name]) return variables[name];
    if (fallback) return fallback.trim();
    unresolved = true;
    return "";
  });
  if (unresolved) return null;
  // pxtransform 在插件之前跑完，内联进来的 px 还没被减半，这里补上，否则尺寸翻倍
  return next.replace(DECLARED_PX, (_, n) => `${parseFloat(n) / 2}px`);
}

function convertUnits(value) {
  return CONVERTERS.reduce(
    (acc, [pattern, fn]) =>
      acc.replace(
        pattern,
        (_, n) => `${Number(parseFloat(fn(parseFloat(n))).toFixed(3))}px`
      ),
    value
  );
}

// 现代空格分隔写法 rgb(0 0 0 / 4%) 不被 css-to-react-native 识别，转成传统逗号写法
function normalizeColors(value) {
  return value.replace(
    /rgba?\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(?:\/\s*([\d.]+)(%?))?\s*\)/g,
    (_, r, g, b, a, pct) => {
      if (a === undefined) return `rgb(${r}, ${g}, ${b})`;
      const alpha = pct ? (parseFloat(a) / 100).toFixed(3) : a;
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  );
}

// 这些属性上的 none 在 RN 里等价于「不设置」，css-to-react-native 又会解析失败，删掉最接近原意。
// display / flex 上的 none 有真实语义（隐藏、不伸缩），不能进这个名单。
const NONE_IS_ABSENT =
  /^(border(-(top|right|bottom|left|width|color|style))?|background(-image)?|box-shadow|text-shadow|outline|text-decoration)$/;

// 带 % 的 transform 与 font-family 字体族列表 css-to-react-native 不支持，
// 前者需要改成具体 dp 位移，后者用系统字体替代。
// RN 里没有对应概念、且会被当作非法 style key 透传给原生的属性（会在页面上弹出黄框警告）
const UNSUPPORTED = new Set([
  "white-space", // 换行由 Text 的 numberOfLines 控制
  "text-overflow", // 同上，需要 ellipsizeMode
  "box-sizing", // RN 恒为 border-box
  "cursor",
  "user-select",
  "word-break",
  "vertical-align",
  "outline",
  "outline-offset",
]);

function shouldDrop(decl) {
  const { prop, value } = decl;
  if (UNSUPPORTED.has(prop)) return true;
  if (value === "none") return NONE_IS_ABSENT.test(prop);
  if (value === "inherit" || value === "initial" || value === "unset")
    return true;
  if (prop === "font-family") return true;
  if (prop === "transform" && value.includes("%")) return true;
  return false;
}

/**
 * RN 没有后代选择器，css-to-react-native 对含空格的选择器直接跳过整条规则。
 * 这里把 `.parent .child`、`.parent > .child` 收敛成最后一个类名 —— 业务代码的 className
 * 本来就写在对应元素上，收敛后能直接命中，不需要改 tsx。
 *
 * 伪类（:active 等）与属性选择器不能收敛：把 `.btn:active` 变成 `.btn` 会把按压态变成常态。
 */
function flattenSelector(selector) {
  const value = selector.trim();
  if (/[:[+~]/.test(value)) return null;
  const last = value.split(/\s+/).pop();
  return /^\.[\w-]+$/.test(last) ? last : null;
}

module.exports = () => ({
  postcssPlugin: "postcss-rn-compat",
  // RN 没有媒体查询，且项目的 Level 4 语法（width <= 768px）会让 css-mediaquery 直接抛错；
  // 关键帧动画同样没有对应机制，整段删除避免留下无法解析的声明
  AtRule(atRule) {
    if (atRule.name === "media" || atRule.name.endsWith("keyframes")) {
      atRule.remove();
    }
  },
  Rule(rule) {
    const kept = [];
    for (const selector of rule.selectors) {
      const flat = flattenSelector(selector);
      if (flat) kept.push(flat);
    }
    if (kept.length === 0) {
      rule.remove();
      return;
    }
    rule.selector = kept.join(", ");

    // 渐变文字靠 background-clip: text + color: transparent 实现，RN 侧由 Text 包装组件
    // 用 MaskedView + LinearGradient 还原，这里只把 background-clip 改名成 RN 可承载的属性名
    rule.walkDecls((decl) => {
      if (decl.prop.startsWith("background-clip")) {
        decl.prop = "backgroundClip";
      }
    });
  },
  Declaration(decl) {
    // 必须最先清掉：postcss 把它存成独立标记，序列化时会再拼回值里，
    // 而 css-to-react-native 只 replace 一次 !import，会留下 "ant" 导致解析失败
    if (decl.important) decl.important = false;

    // 先内联再判断，否则变量值本身是 none / 含 % 时会漏过下面的拦截
    if (decl.value.includes("var(")) {
      const inlined = inlineVariables(decl.value);
      if (inlined === null) {
        decl.remove();
        return;
      }
      decl.value = inlined;
    }
    if (shouldDrop(decl)) {
      decl.remove();
      return;
    }
    // 安全区域要靠 SafeAreaView，calc 需要运行时计算，两者在 RN 静态样式里都无法表达
    if (UNRESOLVABLE.test(decl.value)) {
      decl.remove();
      return;
    }
    // RN 的 borderRadius 只接数字，百分比会以字符串下发到原生并抛
    // "String cannot be cast to java.lang.Double"。项目里 35 处 50% 用于圆形元素，
    // 换成足够大的圆角在正方形元素上等价
    if (decl.prop.includes("radius") && decl.value.includes("%")) {
      decl.value = decl.value.replace(/[\d.]+%/g, "9999px");
    }
    // RN 的 display 只有 flex 与 none，块级 / 行内布局概念不存在（129 处）。
    // 顺带记住主轴方向：flex / inline-flex / grid 的子项横向排列，block 类纵向堆叠
    // 幂等标记：append 之后 postcss 会重新访问这条声明，此时值已是 flex，
    // 不加标记会把 block 也误判成横排
    if (
      decl.prop === "display" &&
      decl.value !== "none" &&
      !decl.__rnDisplayHandled
    ) {
      decl.__rnDisplayHandled = true;
      // CSS 里 flex / grid 的主轴默认是 row，RN 是 column。不补齐的话所有横排布局
      // （按钮图文、列表行、卡片头部）都会变成竖排。
      const rowLike = /^(inline-)?(flex|grid)$/.test(decl.value);
      if (decl.value !== "flex") decl.value = "flex";
      if (rowLike) {
        let hasDirection = false;
        decl.parent.walkDecls((node) => {
          if (node.prop === "flex-direction") hasDirection = true;
        });
        if (!hasDirection) {
          decl.parent.append({ prop: "flex-direction", value: "row" });
        }
      }
    }
    if (GRADIENT.test(decl.value)) {
      // 渐变必须保留原值：RN 侧由 src/platform/ui 的 View / Text 用 LinearGradient 渲染。
      // 改名为 background-image 是为了让 css-to-react-native 原样透传而不是报错。
      decl.prop = "background-image";
      return;
    }
    if (/rgba?\(/.test(decl.value)) {
      decl.value = normalizeColors(decl.value);
    }
    decl.value = convertUnits(decl.value);
  },
});

module.exports.postcss = true;
