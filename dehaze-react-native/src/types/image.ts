/**
 * 图像选择类型（用于路由参数与页面间传递）
 */

export interface SelectedImage {
  /** 唯一标识 */
  id?: string;
  /** 图片 URL（本地文件 URI 或远程 URL） */
  url: string;
  /**
   * GT 参考图 URL（清晰图/clean 版本）
   * 仅样例图片（source='sample'）有值，来自数据集 clearImage；
   * 用户上传/拍照的图片无 GT 参考，此字段为 undefined。
   * 用于指标评估时作为 gtUrl，禁止用 hazy 原图或结果图本身。
   */
  cleanUrl?: string;
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
  /**
   * 历史记录复用：原处理任务使用的算法 ID
   * 仅 source='history' 且该历史记录有 algorithmId 时有值；
   * 用于复用历史记录时直接进入处理页，跳过算法选择。
   */
  algorithmId?: number;
  /**
   * 历史记录复用：原处理任务使用的算法参数（JSON 字符串）
   * 与 algorithmId 同时存在，用于复用历史参数。
   */
  algorithmParams?: string;
  /** 样例图片附加信息（仅 source='sample' 时存在） */
  sampleInfo?: {
    sceneType?: string;
  };
}
