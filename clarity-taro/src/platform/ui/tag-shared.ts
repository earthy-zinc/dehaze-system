export interface TagProps {
  color?: "default" | "primary" | "info" | "success" | "warning" | "danger";
  size?: "small" | "medium" | "large";
  variant?: "outlined" | "contained";
  className?: string;
  children?: React.ReactNode;
  onClick?: () => void;
}
