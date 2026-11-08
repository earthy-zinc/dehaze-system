const path = require("path");
const { getDefaultConfig, mergeConfig } = require("@react-native/metro-config");
const { getMetroConfig } = require("@tarojs/rn-supporter");

const projectRoot = __dirname;
// pnpm workspace：依赖实体在仓库根 node_modules/.pnpm，本包 node_modules 里只有软链
const workspaceRoot = path.resolve(projectRoot, "..");

const defaultConfig = getDefaultConfig(projectRoot);

// components-rn 依赖的 @ant-design/react-native 会带进一份 react 19，与 RN renderer 用的 18.3.1
// 各自持有 dispatcher，组件调用 hooks 时读到的 current 是 null（表现为 useState of null）。
const REACT_ROOT = path.resolve(projectRoot, "node_modules/react");
const PINNED_REACT = {
  react: path.join(REACT_ROOT, "index.js"),
  "react/jsx-runtime": path.join(REACT_ROOT, "jsx-runtime.js"),
  "react/jsx-dev-runtime": path.join(REACT_ROOT, "jsx-dev-runtime.js"),
};

module.exports = (async function () {
  const taroConfig = await getMetroConfig();
  const taroResolveRequest =
    taroConfig.resolver && taroConfig.resolver.resolveRequest;

  const config = {
    // 只纳入依赖实体与 workspace 内的 SDK 源码。整仓作为 watchFolders 会把 20 个兄弟项目
    // 和 models 目录一起建索引（约 15 倍包体积），首次构建与内存都会明显变差。
    watchFolders: [
      path.resolve(workspaceRoot, "node_modules"),
      path.resolve(workspaceRoot, "clarity-sdk-js"),
    ],
    resolver: {
      unstable_enableSymlinks: true,
      nodeModulesPaths: [
        path.resolve(projectRoot, "node_modules"),
        path.resolve(workspaceRoot, "node_modules"),
      ],
      // 必须转交给 Taro 自己的 resolver：它负责把入口里的 ./src/app 映射到项目源码，
      // 直接调 context.resolveRequest 会让这个相对路径相对 rn-supporter 包解析而失败
      resolveRequest: (context, moduleName, platform) => {
        const pinned = PINNED_REACT[moduleName];
        if (pinned) {
          return { type: "sourceFile", filePath: pinned };
        }
        // @/ 别名：webpack 侧由 tsconfig-paths 处理，Metro 侧需要显式映射成绝对路径，
        // 再交给 Taro 的 resolver 去做平台后缀查找（Xxx.rn.tsx 优先）
        if (moduleName.startsWith("@/")) {
          const absolute = path.join(projectRoot, "src", moduleName.slice(2));
          if (typeof taroResolveRequest === "function") {
            return taroResolveRequest(context, absolute, platform);
          }
          return context.resolveRequest(context, absolute, platform);
        }
        // Taroify 的组件在 RN 端没有样式（它的 scss 走不通），换成同目录的实现。
        // babel-plugin-import 会把具名导入改写成 @taroify/core/Button 这种形式。
        // 排除包装组件自身，避免自引用。
        if (!context.originModulePath.includes("src/platform/ui/")) {
          const replaced = {
            "@taroify/core/Button": "TaroifyButton.rn.tsx",
            "@taroify/core/Input": "TaroifyInput.rn.tsx",
            "@taroify/core/Switch": "TaroifySwitch.rn.tsx",
          }[moduleName];
          if (replaced) {
            return {
              type: "sourceFile",
              filePath: path.resolve(projectRoot, "src/platform/ui", replaced),
            };
          }
        }
        if (typeof taroResolveRequest === "function") {
          return taroResolveRequest(context, moduleName, platform);
        }
        return context.resolveRequest(context, moduleName, platform);
      },
      // Taroify 组件内部 import 的 scss 走 Dart Sass 的 @forward 语法，RN 的 sass 编译链解析不了。
      // 这里把 scss 当作静态资源，import 得到资源 id，不参与样式转换。
      // shortcut: 该规则对所有 scss 生效，项目一旦引入自己的 scss 会静默失效（不报错、无样式）；
      // 届时改成在 resolveRequest 里只拦截 @taroify 的样式路径。
      assetExts: [...defaultConfig.resolver.assetExts, "scss"],
    },
  };

  return mergeConfig(defaultConfig, taroConfig, config);
})();
