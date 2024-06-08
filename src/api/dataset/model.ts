/**
 * 数据集查询参数类型
 */
export interface DatasetQuery {
  keywords?: string;
}

/**
 * 数据集
 */
export interface Dataset {
  /**
   * 数据集ID
   */
  id: number;
  /**
   * 父数据集ID
   */
  parentId: number;
  /**
   * 数据集类型
   */
  type: string;
  /**
   * 数据集名称
   */
  name: string;
  /**
   * 数据集描述
   */
  description?: string;
  /**
   * 存储位置
   */
  path: string;
  /**
   * 占用空间大小
   */
  size: string;
  /**
   * 数据项数量（简单理解为图片数量）
   */
  total: number;
  /**
   * 子数据集
   */
  children?: Dataset[];
  /**
   * 创建时间
   */
  createTime?: Date;
  /**
   * 修改时间
   */
  updateTime?: Date;
  /**
   * 状态(1:启用；0:禁用)
   */
  status?: number;
}

export interface ImageItem {
  id: number;
  imgUrl: ImageUrl[];
}

export interface ImageUrl {
  id: number;
  /**
   * 图片类型（有雾图像、无雾图像）
   */
  type: string;
  /**
   * 图片URL
   */
  url: string;
  /**
   * 高清图片URL
   */
  originUrl?: string;
  /**
   * 描述
   */
  description?: string;
}
