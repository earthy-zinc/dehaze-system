import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { FileInfo, ImageFileInfo } from '../model/FileInfo';

/**
 * 文件相关API服务接口
 */
export class FileAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 文件上传检查
   *
   * @param md5 文件md5
   * @returns 检查结果
   */
  public static async uploadCheck(md5: string): Promise<Result<any>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<any>>(`${FileAPI.BASE_URL}/files/check`, { md5: md5 });
  }

  /**
   * 上传文件
   *
   * @param file 文件对象
   * @param modelId 模型ID（可选）
   * @returns 文件信息
   */
  public static async upload(file: File, modelId?: number): Promise<Result<FileInfo | ImageFileInfo>> {
    const formData = new FormData();
    if (modelId) {
      formData.append('modelId', modelId.toString());
    }
    formData.append('file', file);
    
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<FileInfo | ImageFileInfo>>(`${FileAPI.BASE_URL}/files`, formData);
  }

  /**
   * 删除文件
   *
   * @param filePath 文件完整路径
   * @returns 操作结果
   */
  public static async deleteByPath(filePath?: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${FileAPI.BASE_URL}/files`, { filePath: filePath });
  }
}