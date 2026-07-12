/**
 * 文件信息模型
 */
export class FileInfo {
  /** 文件 ID */
  id: number = 0;
  /** 文件名称 */
  name: string = '';
  /** 文件路径 */
  path: string = '';
  /** 文件 URL */
  url: string = '';
  /** 文件大小 */
  size?: string = undefined;
  /** 文件 MD5 */
  md5?: string = undefined;
  /** 对象存储名 */
  objectName?: string = undefined;
  /** 文件类型 */
  type?: string = undefined;
  createTime?: string = undefined;
  updateTime?: string = undefined;
}
