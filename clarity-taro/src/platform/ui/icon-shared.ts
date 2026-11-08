/** 图标组件两端的共用类型，避免平台文件互相导入形成循环 */
export interface IconProps {
  name: string;
  size?: string | number;
  color?: string;
  className?: string;
  style?: Record<string, unknown>;
  onClick?: () => void;
}
