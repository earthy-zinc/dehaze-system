/**
 * 共享弹窗工具函数
 */

import Taro from "@tarojs/taro";

interface ConfirmOptions {
  title?: string;
  content: string;
  confirmText?: string;
  cancelText?: string;
  confirmColor?: string;
}

/**
 * 确认对话框，返回用户是否确认
 * 统一封装 Taro.showModal 的 Promise 化调用
 */
export function confirmDialog(options: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    Taro.showModal({
      title: options.title || "提示",
      content: options.content,
      confirmText: options.confirmText,
      cancelText: options.cancelText,
      confirmColor: options.confirmColor,
      success: (res) => resolve(res.confirm),
      fail: () => resolve(false),
    });
  });
}
