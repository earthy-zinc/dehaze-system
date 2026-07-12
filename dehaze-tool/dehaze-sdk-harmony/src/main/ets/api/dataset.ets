import { HttpManager } from '../utils/HttpManager';
import { PageResult } from '../model/PageResult';
import {
  Dataset,
  DatasetQuery,
  DatasetItem,
  DatasetItemQuery,
  PairedImageUpload,
  DatasetImageFileInfo
} from '../model/DatasetInfo';

/**
 * 数据集管理 API
 *
 * 路径与后端路由对齐：
 * - 数据集：dehaze-go/internal/router/dataset.go → /api/v1/datasets
 * - 数据项：dehaze-go/internal/router/dataset_item.go → /api/v1/dataset-items
 * - 图片文件：dehaze-go/internal/router/item_file.go → /api/v1/item-files
 */
export class DatasetAPI {
  private static readonly BASE_URL: string = '/api/v1';

  // ==================== 数据集接口 ====================

  /** 获取数据集列表（树形） */
  public static async getDatasetList(query?: DatasetQuery): Promise<Dataset[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Dataset[]>(`${DatasetAPI.BASE_URL}/datasets`, query as unknown as Record<string, Object>);
  }

  /** 获取数据集树形结构 */
  public static async getDatasetTree(): Promise<Dataset[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Dataset[]>(`${DatasetAPI.BASE_URL}/datasets/tree`, undefined);
  }

  /** 获取数据集下拉选项 */
  public static async getOptions(): Promise<Array<{ value: number; label: string }>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Array<{ value: number; label: string }>>(`${DatasetAPI.BASE_URL}/datasets/options`, undefined);
  }

  /** 获取数据集详情 */
  public static async getDatasetById(id: number): Promise<Dataset> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Dataset>(`${DatasetAPI.BASE_URL}/datasets/${id}`, undefined);
  }

  /** 新增数据集 */
  public static async add(data: Dataset): Promise<number> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<number>(`${DatasetAPI.BASE_URL}/datasets`, data as unknown as Record<string, Object>);
  }

  /** 修改数据集 */
  public static async update(id: number, data: Dataset): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.put<void>(`${DatasetAPI.BASE_URL}/datasets/${id}`, data as unknown as Record<string, Object>);
  }

  /** 删除数据集 */
  public static async delete(id: number): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${DatasetAPI.BASE_URL}/datasets/${id}`, undefined);
  }

  /** 批量删除数据集 */
  public static async batchDelete(ids: number[]): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${DatasetAPI.BASE_URL}/datasets/batch`, { ids } as Record<string, Object>);
  }

  // ==================== 数据项接口 ====================

  /** 分页查询数据项列表 */
  public static async getDatasetItems(query: DatasetItemQuery): Promise<PageResult<DatasetItem>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PageResult<DatasetItem>>(`${DatasetAPI.BASE_URL}/dataset-items`, query as unknown as Record<string, Object>);
  }

  /** 获取数据项详情 */
  public static async getDatasetItemById(id: number): Promise<DatasetItem> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<DatasetItem>(`${DatasetAPI.BASE_URL}/dataset-items/${id}`, undefined);
  }

  /** 创建空数据项 */
  public static async createDatasetItem(datasetId: number, name?: string): Promise<number> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<number>(`${DatasetAPI.BASE_URL}/dataset-items`, { datasetId, name } as Record<string, Object>);
  }

  /**
   * 创建数据项并上传配对图片
   *
   * 使用 multipart/form-data，需通过文件 URI 上传。
   * 后端路径：POST /api/v1/dataset-items/upload
   */
  public static async uploadWithImages(data: PairedImageUpload): Promise<DatasetImageFileInfo> {
    const httpManager = HttpManager.getInstance();
    const formData: Record<string, string> = {
      datasetId: data.datasetId.toString(),
    };
    if (data.description) {
      formData.description = data.description;
    }
    // 上传清晰图
    if (data.clearImageUri) {
      return await httpManager.upload<DatasetImageFileInfo>({
        url: `${DatasetAPI.BASE_URL}/dataset-items/upload`,
        fileUri: data.clearImageUri,
        name: 'clearImage',
        formData,
      });
    }
    // 上传有雾图
    if (data.hazyImageUri) {
      return await httpManager.upload<DatasetImageFileInfo>({
        url: `${DatasetAPI.BASE_URL}/dataset-items/upload`,
        fileUri: data.hazyImageUri,
        name: 'hazyImage',
        formData,
      });
    }
    throw new Error('必须提供清晰图或有雾图');
  }

  /** 修改数据项 */
  public static async updateDatasetItem(id: number, data: Partial<DatasetItem>): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.put<void>(`${DatasetAPI.BASE_URL}/dataset-items/${id}`, data as unknown as Record<string, Object>);
  }

  /** 删除数据项 */
  public static async deleteDatasetItem(id: number): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${DatasetAPI.BASE_URL}/dataset-items/${id}`, undefined);
  }

  /** 批量删除数据项 */
  public static async batchDeleteDatasetItems(ids: number[]): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${DatasetAPI.BASE_URL}/dataset-items/batch`, { ids } as Record<string, Object>);
  }

  // ==================== 图片文件接口 ====================

  /** 获取图片详细信息 */
  public static async getItemFileById(id: number): Promise<DatasetImageFileInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<DatasetImageFileInfo>(`${DatasetAPI.BASE_URL}/item-files/${id}`, undefined);
  }

  /** 上传数据项图片 */
  public static async uploadItemImage(fileUri: string, formData?: Record<string, string>): Promise<DatasetImageFileInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.upload<DatasetImageFileInfo>({
      url: `${DatasetAPI.BASE_URL}/item-files`,
      fileUri,
      name: 'file',
      formData,
    });
  }

  /** 修改图片信息 */
  public static async updateItemImage(id: number, type: string, description?: string): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.put<void>(`${DatasetAPI.BASE_URL}/item-files/${id}`, { type, description } as Record<string, Object>);
  }

  /** 删除图片 */
  public static async deleteItemImage(id: number): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${DatasetAPI.BASE_URL}/item-files/${id}`, undefined);
  }

  /** 批量删除图片 */
  public static async batchDeleteItemImages(ids: number[]): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${DatasetAPI.BASE_URL}/item-files/batch`, { ids } as Record<string, Object>);
  }
}
