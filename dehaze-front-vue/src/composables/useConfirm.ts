import { ElMessageBox } from "element-plus";

/**
 * 确认弹窗通用配置。
 */
interface ConfirmOptions {
  title?: string;
  message?: string;
  confirmButtonText?: string;
  cancelButtonText?: string;
  type?: "warning" | "error" | "info" | "success";
  distinguishCancelAndClose?: boolean;
}

/**
 * 统一默认配置。
 */
const DEFAULT_OPTIONS: Required<Omit<ConfirmOptions, "title" | "message">> = {
  confirmButtonText: "确定",
  cancelButtonText: "取消",
  type: "warning",
  distinguishCancelAndClose: true,
};

/**
 * 二次确认弹窗 Composable，基于 Element Plus ElMessageBox.confirm 封装。
 *
 * @example
 * ```ts
 * const confirm = useConfirm();
 *
 * // 基础用法
 * const ok = await confirm('确定要执行此操作吗？');
 *
 * // 自定义标题和内容
 * const ok = await confirm('这是一条重要提示', '系统通知', { type: 'info' });
 *
 * // 快捷删除确认
 * const ok = await useDeleteConfirm('用户');
 * ```
 */
export function useConfirm() {
  /**
   * 弹出确认对话框。
   *
   * @param message - 提示信息
   * @param title - 标题
   * @param options - 额外配置项
   * @returns Promise<boolean> - 用户点击确认返回 true，取消返回 false
   */
  const confirm = async (
    message?: string,
    title?: string,
    options?: ConfirmOptions
  ): Promise<boolean> => {
    try {
      const {
        message: msg = message,
        confirmButtonText = DEFAULT_OPTIONS.confirmButtonText,
        cancelButtonText = DEFAULT_OPTIONS.cancelButtonText,
        type = DEFAULT_OPTIONS.type,
        distinguishCancelAndClose = DEFAULT_OPTIONS.distinguishCancelAndClose,
      } = options ?? {};

      await ElMessageBox.confirm(
        msg ?? "确定要执行此操作吗？",
        title ?? "提示",
        {
          confirmButtonText,
          cancelButtonText,
          type,
          distinguishCancelAndClose,
          ...(options ?? {}),
        }
      );
      return true;
    } catch (error: unknown) {
      // 用户取消或关闭弹窗
      return false;
    }
  };

  return confirm;
}

/**
 * 快捷删除确认，生成"确定删除 xxx 吗？"的标准提示。
 *
 * @param itemName - 待删除项的名称
 * @returns Promise<boolean> - 用户点击确认返回 true，取消返回 false
 *
 * @example
 * ```ts
 * const ok = await useDeleteConfirm('该用户');
 * if (ok) {
 *   // 执行删除逻辑
 * }
 * ```
 */
export function useDeleteConfirm(itemName?: string): Promise<boolean> {
  const message = itemName
    ? `确定删除${itemName}吗？`
    : "确定要删除所选内容吗？";

  return ElMessageBox.confirm(message, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    distinguishCancelAndClose: true,
  })
    .then(() => true)
    .catch(() => false);
}
