/**
 * 图像选择类型（用于路由参数与页面间传递）
 */

export interface SelectedImage {
  /** 唯一标识 */
  id?: string;
  /** 图片 URL（本地文件 URI 或远程 URL） */
  url: string;
  /** 缩略图 URL */
  thumbUrl?: string;
  /** 图片名称 */
  name?: string;
  /** 宽度 */
  width?: number;
  /** 高度 */
  height?: number;
  /** 文件大小（字节） */
  size?: number;
  /** 来源 */
  source?: 'upload' | 'camera' | 'sample' | 'history';
  /** 样例图片附加信息（仅 source='sample' 时存在） */
  sampleInfo?: {
    sceneType?: string;
  };
}
