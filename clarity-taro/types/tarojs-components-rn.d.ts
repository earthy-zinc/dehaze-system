// components-rn 未提供类型声明；RN 端的 View / Text 包装只需要透传 props，
// 这里给最小声明避免引入 any 扩散。
declare module "@tarojs/components-rn" {
  import type { ComponentType, ReactNode } from "react";

  type AnyProps = {
    style?: unknown;
    className?: string;
    children?: ReactNode;
    [key: string]: unknown;
  };

  export const View: ComponentType<AnyProps>;
  export const Text: ComponentType<AnyProps>;
  export const Button: ComponentType<AnyProps>;
}
