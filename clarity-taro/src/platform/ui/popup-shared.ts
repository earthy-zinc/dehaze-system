export interface PopupProps {
  open?: boolean;
  placement?: "top" | "bottom" | "left" | "right" | "center";
  rounded?: boolean;
  onClose?: (value: never) => void;
  className?: string;
  style?: Record<string, unknown>;
  children?: React.ReactNode;
}
