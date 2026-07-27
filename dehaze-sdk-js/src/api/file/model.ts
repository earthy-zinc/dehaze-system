import { PageQuery } from "@/types";

export interface FileQuery extends PageQuery {
  keywords?: string;
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
