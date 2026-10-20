import type { Logger } from "./Logger";

/**
 * React 错误边界工厂。SDK 自身不依赖 React，React 实例由宿主项目注入：
 * - `Logger.install({ ..., react })` 注入后，使用默认导出的 `ErrorBoundary`。
 * - 也可用 `createErrorBoundary(React, logger)` 显式绑定。
 * 捕获到的渲染错误会交给 Logger 以 ERROR 级别上报（error_type=js）。
 */
export function createErrorBoundary(
  react: { Component: any; createElement: (...args: any[]) => any },
  logger: Logger
) {
  const { Component } = react;

  return class ErrorBoundary extends Component<{
    children: unknown;
    fallbackRender?: (error: Error) => unknown;
  }> {
    state: { error?: Error } = {};

    static getDerivedStateFromError(error: Error) {
      return { error };
    }

    componentDidCatch(error: Error, info: { componentStack?: string }) {
      logger.error(`React 组件渲染异常: ${error.message}`, {
        error_type: "js",
        error_source: "react_error_boundary",
        error_stack: `${error.stack ?? ""}\nComponent Stack:\n${info.componentStack ?? ""}`,
      });
    }

    render() {
      if (this.state.error) {
        return this.props.fallbackRender ? this.props.fallbackRender(this.state.error) : null;
      }
      return this.props.children;
    }
  };
}
