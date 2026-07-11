/**
 * 文件API类型声明
 */

/**
 * 文件查询参数
 */
export interface FileQuery {
  /** 关键字（文件名或文件类型模糊匹配） */
  keywords?: string;
  /** 页码，默认 1 */
  pageNum?: number;
  /** 每页数量，默认 10 */
  pageSize?: number;
}

/**
 * 文件信息
 */
export interface FileInfo {
  /** 文件 ID */
  id: number;
  /** 文件原始名称 */
  name: string;
  /** 文件类型（扩展名） */
  type?: string;
  /** 文件大小（格式化显示，如 "2.44MB"） */
  size?: string;
  /** 文件大小（原始字节数） */
  sizeBytes?: number;
  /** 文件访问 URL */
  url: string;
  /** 文件存储路径 */
  path: string;
  /** 文件存储对象名 */
  objectName?: string;
  /** 文件 MD5 值 */
  md5?: string;
  /** 创建时间 */
  createTime?: string;
}
