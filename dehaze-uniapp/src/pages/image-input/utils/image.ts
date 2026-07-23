/**
 * 图像处理共享工具函数
 */

/** 获取图片信息（宽高） */
export function getImageInfo(
  src: string
): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    uni.getImageInfo({
      src,
      success: (res) => resolve({ width: res.width, height: res.height }),
      fail: reject,
    });
  });
}
