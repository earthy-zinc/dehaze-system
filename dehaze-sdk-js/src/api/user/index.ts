import { PageResult } from "@/types";
import request from "@/utils/request";
import { UserForm, UserInfo, UserPageVO, UserQuery } from "./model";

class UserAPI {
  /**
   * 登录成功后获取用户信息（昵称、头像、权限集合和角色集合）
   */
  static getInfo() {
    return request<any, UserInfo>({
      url: "/api/v1/auth/me",
      method: "get",
    });
  }

  /**
   * 获取用户分页列表
   *
   * @param queryParams
   */
  static getPage(queryParams: UserQuery) {
    return request<any, PageResult<UserPageVO[]>>({
      url: "/api/v1/users/page",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 获取用户表单详情
   *
   * @param userId
   */
  static getFormData(userId: number) {
    return request<any, UserForm>({
      url: "/api/v1/users/" + userId + "/form",
      method: "get",
    });
  }

  /**
   * 添加用户
   *
   * @param data
   */
  static add(data: UserForm) {
    return request({
      url: "/api/v1/users",
      method: "post",
      data: data,
    });
  }

  /**
   * 修改用户
   *
   * @param id
   * @param data
   */
  static update(id: number, data: UserForm) {
    return request({
      url: "/api/v1/users/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 修改用户密码（PATCH，请求体传参）
   *
   * @param id 用户ID
   * @param password 新密码
   */
  static updatePassword(id: number, password: string) {
    return request({
      url: "/api/v1/users/" + id + "/password",
      method: "patch",
      data: { password },
    });
  }

  /**
   * 修改用户状态（PATCH，query param）
   *
   * @param id 用户ID
   * @param status 状态(1-启用;0-禁用)
   */
  static updateStatus(id: number, status: number) {
    return request({
      url: "/api/v1/users/" + id + "/status",
      method: "patch",
      params: { status },
    });
  }

  /**
   * 删除用户（路径参数，逗号分隔）
   *
   * @param ids 用户ID字符串，多个以英文逗号(,)分割
   */
  static deleteByIds(ids: string) {
    return request({
      url: "/api/v1/users/" + ids,
      method: "delete",
    });
  }

  /**
   * 下载用户导入模板
   */
  static downloadTemplate() {
    return request({
      url: "/api/v1/users/template",
      method: "get",
      responseType: "arraybuffer",
    });
  }

  /**
   * 导出用户
   *
   * @param queryParams
   */
  static export(queryParams?: Partial<UserQuery>) {
    return request({
      url: "/api/v1/users/_export",
      method: "get",
      params: queryParams,
      responseType: "arraybuffer",
    });
  }

  /**
   * 导入用户（multipart/form-data）
   *
   * @param deptId 部门ID
   * @param file Excel文件
   */
  static import(deptId: number, file: File) {
    const formData = new FormData();
    formData.append("file", file);
    return request({
      url: "/api/v1/users/_import",
      method: "post",
      params: { deptId },
      data: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }
}

export default UserAPI;
