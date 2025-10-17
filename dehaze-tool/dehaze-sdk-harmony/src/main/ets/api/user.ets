import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { PageResult } from '../model/PageResult';
import { UserInfo, UserQuery, UserPageVO, UserForm } from '../model/UserInfo';

/**
 * 用户相关API服务接口
 */
export class UserAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 登录成功后获取用户信息（昵称、头像、权限集合和角色集合）
   */
  public static async getInfo(): Promise<Result<UserInfo>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<UserInfo>>(`${UserAPI.BASE_URL}/users/me`);
  }

  /**
   * 获取用户分页列表
   *
   * @param queryParams 查询参数
   */
  public static async getPage(queryParams: UserQuery): Promise<Result<PageResult<Array<UserPageVO>>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<PageResult<Array<UserPageVO>>>>(`${UserAPI.BASE_URL}/users/page`, queryParams);
  }

  /**
   * 获取用户表单详情
   *
   * @param userId 用户ID
   */
  public static async getFormData(userId: number): Promise<Result<UserForm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<UserForm>>(`${UserAPI.BASE_URL}/users/${userId}/form`);
  }

  /**
   * 添加用户
   *
   * @param data 用户表单数据
   */
  public static async add(data: UserForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${UserAPI.BASE_URL}/users`, data);
  }

  /**
   * 修改用户
   *
   * @param id 用户ID
   * @param data 用户表单数据
   */
  public static async update(id: number, data: UserForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${UserAPI.BASE_URL}/users/${id}`, data);
  }

  /**
   * 修改用户密码
   *
   * @param id 用户ID
   * @param password 新密码
   */
  public static async updatePassword(id: number, password: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.patch<Result<void>>(`${UserAPI.BASE_URL}/users/${id}/password`, { password: password });
  }

  /**
   * 删除用户
   *
   * @param ids 用户ID字符串，多个用逗号分隔
   */
  public static async deleteByIds(ids: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${UserAPI.BASE_URL}/users/${ids}`);
  }

  /**
   * 下载用户导入模板
   *
   * @returns 文件数据
   */
  public static async downloadTemplate(): Promise<ArrayBuffer> {
    // 这个功能在鸿蒙SDK中需要特殊处理
    throw new Error('此功能暂未实现，请使用其他方式下载模板');
  }

  /**
   * 导出用户
   *
   * @param queryParams 查询参数
   * @returns 文件数据
   */
  public static async export(queryParams: UserQuery): Promise<ArrayBuffer> {
    // 这个功能在鸿蒙SDK中需要特殊处理
    throw new Error('此功能暂未实现，请使用其他方式导出用户数据');
  }

  /**
   * 导入用户
   *
   * @param deptId 部门ID
   * @param file 文件对象
   */
  public static async import(deptId: number, file: File): Promise<Result<void>> {
    const formData = new FormData();
    formData.append('file', file);
    
    // 这个功能在鸿蒙SDK中需要特殊处理
    throw new Error('此功能暂未实现，请使用其他方式导入用户数据');
  }
}