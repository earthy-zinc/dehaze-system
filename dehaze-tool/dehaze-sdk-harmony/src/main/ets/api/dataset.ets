import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { PageResult } from '../model/PageResult';
import {
  Dataset,
  DatasetQuery,
  DatasetImageFileInfo,
  ImageItem,
  ImageItemQuery
} from '../model/DatasetInfo';

/**
 * 数据集相关API服务接口
 */
export class DatasetAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 数据集树形表格
   *
   * @param queryParams 查询参数
   * @returns 数据集列表
   */
  public static async getList(queryParams?: DatasetQuery): Promise<Result<Array<Dataset>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<Dataset>>>(`${DatasetAPI.BASE_URL}/dataset`, queryParams);
  }

  /**
   * 获取数据集下拉列表
   *
   * @returns 数据集下拉选项列表
   */
  public static async getOptions(): Promise<Result<Array<any>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<any>>>(`${DatasetAPI.BASE_URL}/dataset/options`);
  }

  /**
   * 根据Id获取数据集信息
   *
   * @param id 数据集ID
   * @returns 数据集详情
   */
  public static async getDatasetInfoById(id: number): Promise<Result<Dataset>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Dataset>>(`${DatasetAPI.BASE_URL}/dataset/${id}`);
  }

  /**
   * 获取数据集详细图片
   *
   * @param id 数据集ID
   * @param queryParams 查询参数
   * @returns 图片项分页列表
   */
  public static async getImageItem(id: number, queryParams: ImageItemQuery): Promise<Result<PageResult<Array<ImageItem>>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<PageResult<Array<ImageItem>>>>(`${DatasetAPI.BASE_URL}/dataset/${id}/images`, queryParams);
  }

  /**
   * 新增数据集
   *
   * @param data 数据集数据
   * @returns 操作结果
   */
  public static async add(data: Dataset): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${DatasetAPI.BASE_URL}/dataset`, data);
  }

  /**
   * 修改数据集
   *
   * @param id 数据集ID
   * @param data 数据集数据
   * @returns 操作结果
   */
  public static async update(id: number, data: Dataset): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${DatasetAPI.BASE_URL}/dataset/${id}`, data);
  }

  /**
   * 删除数据集
   *
   * @param ids 数据集ID数组
   * @returns 操作结果
   */
  public static async deleteByIds(ids: Array<string>): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${DatasetAPI.BASE_URL}/dataset/${ids.join(',')}`);
  }

  /**
   * 新增数据项
   *
   * @param datasetId 数据集ID
   * @param name 数据项名称
   * @returns 数据项ID
   */
  public static async addDatasetItem(datasetId: number, name?: string): Promise<Result<number>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<number>>(`${DatasetAPI.BASE_URL}/dataset/item`, {
      datasetId: datasetId,
      name: name
    });
  }

  /**
   * 修改数据项
   *
   * @param datasetItemId 数据项ID
   * @param name 数据项名称
   * @returns 操作结果
   */
  public static async updateDatasetItem(datasetItemId: number, name: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${DatasetAPI.BASE_URL}/dataset/item`, {
      datasetItemId: datasetItemId,
      name: name
    });
  }

  /**
   * 删除数据项
   *
   * @param datasetItemId 数据项ID
   * @returns 操作结果
   */
  public static async deleteDatasetItem(datasetItemId: number): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${DatasetAPI.BASE_URL}/dataset/item`, {
      datasetItemId: datasetItemId
    });
  }

  /**
   * 上传数据项图片
   *
   * @param datasetId 数据集ID
   * @param datasetItemId 数据项ID
   * @param type 图片类型
   * @param file 文件对象
   * @param description 描述
   * @returns 图片文件信息
   */
  public static async uploadItemImage(
    datasetId: number,
    datasetItemId: number,
    type: string,
    file: File,
    description?: string
  ): Promise<Result<DatasetImageFileInfo>> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('datasetId', datasetId.toString());
    formData.append('datasetItemId', datasetItemId.toString());
    formData.append('type', type);
    if (description) {
      formData.append('description', description);
    }

    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<DatasetImageFileInfo>>(`${DatasetAPI.BASE_URL}/dataset/image`, formData);
  }

  /**
   * 修改数据项图片
   *
   * @param itemFileId 图片文件ID
   * @param type 图片类型
   * @param description 描述
   * @returns 操作结果
   */
  public static async updateItemImage(
    itemFileId: number,
    type: string,
    description?: string
  ): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${DatasetAPI.BASE_URL}/dataset/image`, {
      itemFileId: itemFileId,
      type: type,
      description: description
    });
  }

  /**
   * 删除数据项图片
   *
   * @param itemFileId 图片文件ID
   * @returns 操作结果
   */
  public static async deleteItemImage(itemFileId: number): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${DatasetAPI.BASE_URL}/dataset/image`, {
      itemFileId: itemFileId
    });
  }
}