/**
 * 文件API类型声明
 */
export interface FileInfo {
  name: string;
  url: string;
}

/**
 * ImageFileInfo
 */
export interface ImageFileInfo {
  /**
   * 所属数据集id
   */
  datasetId?: number;
  /**
   * 所属文件id
   */
  fileId: number;
  /**
   * 当前图片id
   */
  id?: number;
  /**
   * 所属数据项id
   */
  imageItemId?: number;
  /**
   * 文件名称
   */
  name: string;
  /**
   * 图片类型
   */
  type?: string;
  /**
   * 文件URL
   */
  url: string;
  [property: string]: any;
}
