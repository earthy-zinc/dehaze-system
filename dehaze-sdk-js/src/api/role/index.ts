import { OptionType } from "@/types";
import request from "@/utils/request";
import { RoleForm, RolePageResult, RoleQuery } from "./model";

class RoleAPI {
  /**
   * 获取角色分页数据
   *
   * @param queryParams
   */
  static getPage(queryParams?: RoleQuery) {
    return request<any, RolePageResult>({
      url: "/api/v1/roles/page",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 获取角色下拉数据源
   */
  static getOptions() {
    return request<any, OptionType[]>({
      url: "/api/v1/roles/options",
      method: "get",
    });
  }

  /**
   * 获取角色的菜单ID集合
   *
   * @param roleId
   */
  static getRoleMenuIds(roleId: number) {
    return request<any, number[]>({
      url: "/api/v1/roles/" + roleId + "/menuIds",
      method: "get",
    });
  }

  /**
   * 分配菜单权限给角色（PATCH）
   *
   * @param roleId
   * @param menuIds 菜单ID数组
   */
  static updateRoleMenus(roleId: number, menuIds: number[]) {
    return request({
      url: "/api/v1/roles/" + roleId + "/menus",
      method: "patch",
      data: menuIds,
    });
  }

  /**
   * 获取角色表单数据
   *
   * @param id 角色ID
   */
  static getFormData(id: number) {
    return request<any, RoleForm>({
      url: "/api/v1/roles/" + id + "/form",
      method: "get",
    });
  }

  /**
   * 添加角色
   *
   * @param data
   */
  static add(data: RoleForm) {
    return request({
      url: "/api/v1/roles",
      method: "post",
      data: data,
    });
  }

  /**
   * 更新角色
   *
   * @param id 角色ID
   * @param data
   */
  static update(id: number, data: RoleForm) {
    return request({
      url: "/api/v1/roles/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 更新角色状态（PATCH）
   *
   * @param roleId 角色ID
   * @param status 状态(1-启用;0-禁用)
   */
  static updateStatus(roleId: number, status: number) {
    return request({
      url: "/api/v1/roles/" + roleId + "/status",
      method: "patch",
      params: { status },
    });
  }

  /**
   * 批量删除角色（路径参数，逗号分隔）
   *
   * @param ids 角色ID字符串，多个以英文逗号(,)分割
   */
  static deleteByIds(ids: string) {
    return request({
      url: "/api/v1/roles/" + ids,
      method: "delete",
    });
  }
}

export default RoleAPI;
