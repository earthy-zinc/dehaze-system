/**
 * 数据集查询参数类型
 */
export class DatasetQuery {
  keywords?: string = undefined;
}

/**
 * 数据集
 */
export class Dataset {
  /**
   * 数据集ID
   */
  id: number = 0;
  /**
   * 父数据集ID
   */
  parentId: number = 0;
  /**
   * 数据集类型
   */
  type: string = '';
  /**
   * 数据集名称
   */
  name: string = '';
  /**
   * 数据集描述
   */
  description?: string = undefined;
  /**
   * 存储位置
   */
  path: string = '';
  /**
   * 占用空间大小
   */
  size: string = '';
  /**
   * 数据项数量（简单理解为图片数量）
   */
  total: number = 0;
  /**
   * 子数据集
   */
  children?: Array<Dataset> = undefined;
  /**
   * 创建时间
   */
  createTime?: string = undefined;
  /**
   * 修改时间
   */
  updateTime?: string = undefined;
  /**
   * 状态(1:启用；0:禁用)
   */
  status?: number = undefined;
}

export class ImageItemQuery {
  keywords?: string = undefined;
  pageNum: number = 1;
  pageSize: number = 10;
}

export class ImageItem {
  id: number = 0;
  imgUrl: Array<ImageUrl> = [];
}

export class ImageUrl {
  id: number = 0;
  /**
   * 图片类型（有雾图像、无雾图像）
   */
  type: string = '';
  /**
   * 图片URL
   */
  url: string = '';
  /**
   * 高清图片URL
   */
  originUrl?: string = undefined;
  /**
   * 描述
   */
  description?: string = undefined;
}

export class DatasetImageFileInfo {
  id: number = 0;
  datasetItemId: number = 0;
  fileId: number = 0;
  type: string = '';
  description: string = '';
  url: string = '';
}