/**
 * 共享图片保存工具
 */

import Taro from "@tarojs/taro";
import { getErrorMessage } from "./error";
import { isH5 } from "@/utils/platform";

interface SaveImageOptions {
  /** H5 端是否通过 a 标签触发下载（默认 true） */
  h5Download?: boolean;
}

/**
 * 保存网络图片到相册
 * 小程序/App：先下载到本地临时文件，再保存到系统相册
 * H5：若 h5Download 开启，通过 a 标签触发浏览器下载
 */
export async function saveImageToAlbum(
  url: string,
  options: SaveImageOptions = {}
): Promise<void> {
  const { h5Download = true } = options;

  try {
    const downloadRes = await Taro.downloadFile({ url });
    if (downloadRes.statusCode !== 200) {
      throw new Error("下载结果图片失败");
    }

    if (isH5 && h5Download) {
      // H5 端：通过 a 标签触发下载
      const link = document.createElement("a");
      link.href = downloadRes.tempFilePath;
      link.download = `dehaze-result-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      Taro.showToast({ title: "已开始下载", icon: "success" });
      return;
    }

    await Taro.saveImageToPhotosAlbum({
      filePath: downloadRes.tempFilePath,
    });
    Taro.showToast({ title: "已保存到相册", icon: "success" });
  } catch (error: unknown) {
    const errMsg = (error as { errMsg?: string })?.errMsg;
    if (errMsg?.includes("auth deny") || errMsg?.includes("authorize")) {
      Taro.showModal({
        title: "提示",
        content: "需要相册权限才能保存图片，请在设置中开启",
        confirmText: "去设置",
        success: (res) => {
          if (res.confirm) {
            Taro.openSetting();
          }
        },
      });
    } else {
      Taro.showToast({
        title: getErrorMessage(error, "保存失败"),
        icon: "none",
      });
    }
  }
}
