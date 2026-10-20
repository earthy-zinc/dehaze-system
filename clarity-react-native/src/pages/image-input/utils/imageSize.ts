/**
 * 图片尺寸 fallback 工具
 *
 * 尝试通过 Image.getSize 获取真实尺寸，失败时回退到默认 1920x1080。
 */
import { imageInputApi } from '../services/imageInputApi';

const DEFAULT_WIDTH = 1920;
const DEFAULT_HEIGHT = 1080;

export async function getImageSizeWithFallback(
  uri: string,
): Promise<{ width: number; height: number }> {
  try {
    return await imageInputApi.getImageSize(uri);
  } catch {
    return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
  }
}
