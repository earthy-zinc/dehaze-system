/**
 * 数据集查询参数类型
 */
export interface DatasetQuery {
  keywords?: string;
}

/**
 * 数据集
 */
export interface DatasetVO {
  /**
   * 数据集ID
   */
  id?: number;
  /**
   * 父数据集ID
   */
  parentId?: number;
  /**
   * 数据集类型
   */
  type?: string;
  /**
   * 数据集名称
   */
  name?: string;
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
  children?: DatasetVO[];
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

/**
 * 数据项
 */
export interface DatasetItem {
  /**
   * 数据项ID
   */
  id: number;
  /**
   * 所属数据集ID
   */
  datasetId: number;
  /**
   * 详细数据
   */
  data: ImageData[];
}

export interface ImageData {
  /**
   * 图片类型
   */
  type?: string;
  /**
   * 图片URL
   */
  url?: string;
  /**
   * 图片分辨率
   */
  resolution?: string;
  /**
   * 图片大小
   */
  size?: string;
  /**
   * 存储位置
   */
  path?: string;
}
