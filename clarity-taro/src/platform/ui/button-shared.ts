/** 两套按钮共用的类型与判定，独立成文件：平台文件互相导入会形成循环 */
export interface ButtonProps {
  className?: string;
  style?: Record<string, unknown>;
  onClick?: (event: {
    stopPropagation: () => void;
    [key: string]: unknown;
  }) => void;
  children?: React.ReactNode;
  color?: "default" | "primary" | "info" | "success" | "warning" | "danger";
  size?: "mini" | "small" | "medium" | "large";
  block?: boolean;
  variant?: "outlined" | "contained";
  loading?: boolean;
  formType?: "submit" | "reset";
  disabled?: boolean;
  [key: string]: unknown;
}

/** 是否按 Taroify 的按钮形态渲染 */
export function isTaroifyButton(props: ButtonProps): boolean {
  return (
    props.color !== undefined ||
    props.size !== undefined ||
    props.block !== undefined ||
    props.variant !== undefined ||
    props.loading !== undefined ||
    props.formType !== undefined
  );
}
