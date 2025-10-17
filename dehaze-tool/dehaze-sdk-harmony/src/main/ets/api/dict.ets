import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { PageResult } from '../model/PageResult';
import {
  DictTypeQuery,
  DictTypePageVO,
  DictTypeForm,
  DictQuery,
  DictForm,
  DictPageVO
} from '../model/DictInfo';

/**
 * 字典相关API服务接口
 */
export class DictAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 字典类型分页列表
   *
   * @param queryParams 查询参数
   * @returns 字典类型分页列表
   */
  public static async getDictTypePage(queryParams: DictTypeQuery): Promise<Result<PageResult<Array<DictTypePageVO>>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<PageResult<Array<DictTypePageVO>>>>(`${DictAPI.BASE_URL}/dict/types/page`, queryParams);
  }

  /**
   * 字典类型表单数据
   *
   * @param id 字典类型ID
   * @returns 字典类型表单数据
   */
  public static async getDictTypeForm(id: number): Promise<Result<DictTypeForm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<DictTypeForm>>(`${DictAPI.BASE_URL}/dict/types/${id}/form`);
  }

  /**
   * 新增字典类型
   *
   * @param data 字典类型表单数据
   * @returns 操作结果
   */
  public static async addDictType(data: DictTypeForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${DictAPI.BASE_URL}/dict/types`, data);
  }

  /**
   * 修改字典类型
   *
   * @param id 字典类型ID
   * @param data 字典类型表单数据
   * @returns 操作结果
   */
  public static async updateDictType(id: number, data: DictTypeForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${DictAPI.BASE_URL}/dict/types/${id}`, data);
  }

  /**
   * 删除字典类型
   *
   * @param ids 字典类型ID，多个以英文逗号(,)分割
   * @returns 操作结果
   */
  public static async deleteDictTypes(ids: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${DictAPI.BASE_URL}/dict/types/${ids}`);
  }

  /**
   * 获取字典类型的数据项
   *
   * @param typeCode 字典类型编码
   * @returns 字典项下拉选项列表
   */
  public static async getDictOptions(typeCode: string): Promise<Result<Array<any>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<any>>>(`${DictAPI.BASE_URL}/dict/${typeCode}/options`);
  }

  /**
   * 字典分页列表
   *
   * @param queryParams 查询参数
   * @returns 字典分页列表
   */
  public static async getDictPage(queryParams: DictQuery): Promise<Result<PageResult<Array<DictPageVO>>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<PageResult<Array<DictPageVO>>>>(`${DictAPI.BASE_URL}/dict/page`, queryParams);
  }

  /**
   * 获取字典表单数据
   *
   * @param id 字典ID
   * @returns 字典表单数据
   */
  public static async getDictFormData(id: number): Promise<Result<DictForm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<DictForm>>(`${DictAPI.BASE_URL}/dict/${id}/form`);
  }

  /**
   * 新增字典
   *
   * @param data 字典表单数据
   * @returns 操作结果
   */
  public static async addDict(data: DictForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${DictAPI.BASE_URL}/dict`, data);
  }

  /**
   * 修改字典项
   *
   * @param id 字典ID
   * @param data 字典表单数据
   * @returns 操作结果
   */
  public static async updateDict(id: number, data: DictForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${DictAPI.BASE_URL}/dict/${id}`, data);
  }

  /**
   * 删除字典
   *
   * @param ids 字典项ID，多个以英文逗号(,)分割
   * @returns 操作结果
   */
  public static async deleteDictByIds(ids: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${DictAPI.BASE_URL}/dict/${ids}`);
  }
}