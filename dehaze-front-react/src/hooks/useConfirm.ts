import { Modal, type ButtonProps } from "antd";

interface ConfirmOptions {
  title?: string;
  content: string;
  okText?: string;
  cancelText?: string;
  okType?: "default" | "primary" | "dashed" | "link" | "text" | "danger";
  okButtonProps?: ButtonProps;
}

export function useConfirm() {
  const confirm = (options: ConfirmOptions): Promise<boolean> => {
    return new Promise((resolve) => {
      Modal.confirm({
        title: options.title || "提示",
        content: options.content,
        okText: options.okText || "确定",
        cancelText: options.cancelText || "取消",
        okType: options.okType || "primary",
        okButtonProps: options.okButtonProps,
        onOk: () => resolve(true),
        onCancel: () => resolve(false),
      });
    });
  };

  const deleteConfirm = (itemName?: string): Promise<boolean> => {
    return confirm({
      title: "确认删除",
      content: `确定删除${itemName || "该项"}吗？此操作不可恢复。`,
      okText: "删除",
      okType: "danger",
    });
  };

  return { confirm, deleteConfirm };
}
