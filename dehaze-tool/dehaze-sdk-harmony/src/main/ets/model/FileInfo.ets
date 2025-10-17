/**
 * 文件API类型声明
 */
export class FileInfo {
  name: string = '';
  url: string = '';
}

/**
 * ImageFileInfo
 */
export class ImageFileInfo {
  /**
   * 所属数据集id
   */
  datasetId?: number = undefined;
  /**
   * 所属文件id
   */
  fileId: number = 0;
  /**
   * 当前图片id
   */
  id?: number = undefined;
  /**
   * 所属数据项id
   */
  imageItemId?: number = undefined;
  /**
   * 文件名称
   */
  name: string = '';
  /**
   * 图片类型
   */
  type?: string = undefined;
  /**
   * 文件URL
   */
  url: string = '';
}