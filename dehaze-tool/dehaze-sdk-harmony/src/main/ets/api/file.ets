import { HttpManager } from '../utils/HttpManager';
import { FileInfo } from '../model/FileInfo';

/**
 * 文件管理 API
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/file.go）：
 * - GET    /api/v1/files/check             文件校验（秒传）
 * - GET    /api/v1/files/page              分页查询
 * - GET    /api/v1/files/download/{path}   文件下载
 * - GET    /api/v1/files/{fileId}          文件详情
 * - POST   /api/v1/files                   文件上传
 * - DELETE /api/v1/files                    文件删除
 */
export class FileAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 文件校验（秒传检查） */
  public static async check(md5: string): Promise<FileInfo | null> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<FileInfo | null>(`${FileAPI.BASE_URL}/files/check`, { md5 } as unknown as Record<string, Object>);
  }

  /** 获取文件分页列表 */
  public static async getPage(pageNum: number = 1, pageSize: number = 20, keywords?: string): Promise<{ list: FileInfo[]; total: number }> {
    const httpManager = HttpManager.getInstance();
    const params: Record<string, Object> = { pageNum, pageSize };
    if (keywords) {
      params.keywords = keywords;
    }
    return await httpManager.get<{ list: FileInfo[]; total: number }>(`${FileAPI.BASE_URL}/files/page`, params);
  }

  /** 获取文件详情 */
  public static async getById(fileId: number): Promise<FileInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<FileInfo>(`${FileAPI.BASE_URL}/files/${fileId}`, undefined);
  }

  /**
   * 上传文件
   *
   * @param fileUri 文件 URI（通过 PhotoViewPicker 或相机获取）
   * @param formData 附加表单数据
   * @returns 文件信息
   */
  public static async upload(fileUri: string, formData?: Record<string, string>): Promise<FileInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.upload<FileInfo>({
      url: `${FileAPI.BASE_URL}/files`,
      fileUri,
      name: 'file',
      formData,
    });
  }

  /** 删除文件 */
  public static async delete(filePath?: string): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${FileAPI.BASE_URL}/files`, { filePath } as Record<string, Object>);
  }
}
