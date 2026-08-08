import { Logger, getReact } from "./Logger";
import { createErrorBoundary } from "./ErrorBoundary";

export {
  Logger,
  bindReact,
  defaultStorage,
  generateTraceId,
  getCurrentTraceId,
  getReact,
  setCurrentTraceId,
} from "./Logger";
export { ConsoleTransport, RemoteTransport } from "./transports";
export { createErrorBoundary };
export type {
  LogEntry,
  LogLevel,
  LoggerStorage,
  LogTransport,
  InstallConfig,
} from "./types";

/**
 * React 错误边界组件（对外契约）。依赖宿主通过 `Logger.install({ react })` 注入的 React 实例。
 * 未注入 React 时直接渲染 children（不拦截），避免破坏宿主渲染。
 */
export function ErrorBoundary(props: {
  children: unknown;
  fallbackRender?: (error: Error) => unknown;
}): any {
  const logger = Logger.getInstance();
  const react = getReact();
  if (!logger || !react) {
    return props.children;
  }
  const Bound = createErrorBoundary(react as any, logger);
  return (react as any).createElement(Bound, props);
}
