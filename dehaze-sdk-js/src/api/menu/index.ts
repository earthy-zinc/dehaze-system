import { OptionType } from "@/types";
import request from "@/utils/request";
import { MenuForm, MenuQuery, MenuVO, RouteVO } from "./model";

class MenuAPI {
  /**
   * 获取路由列表
   */
  static getRoutes() {
    return request<any, RouteVO[]>({
      url: "/api/v1/menus/routes",
      method: "get",
    });
  }

  /**
   * 获取菜单树形列表
   *
   * @param queryParams
   */
  static getList(queryParams: MenuQuery) {
    return request<any, MenuVO[]>({
      url: "/api/v1/menus",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 获取菜单下拉数据源
   */
  static getOptions() {
    return request<any, OptionType[]>({
      url: "/api/v1/menus/options",
      method: "get",
    });
  }

  /**
   * 获取菜单表单数据
   *
   * @param id
   */
  static getFormData(id: number) {
    return request<any, MenuForm>({
      url: "/api/v1/menus/" + id + "/form",
      method: "get",
    });
  }

  /**
   * 添加菜单
   *
   * @param data
   * @returns void (后端不返回创建的菜单ID)
   */
  static add(data: MenuForm) {
    return request<any, void>({
      url: "/api/v1/menus",
      method: "post",
      data: data,
    });
  }

  /**
   * 修改菜单
   *
   * @param id
   * @param data
   */
  static update(id: string, data: MenuForm) {
    return request({
      url: "/api/v1/menus/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除菜单（支持批量，路径参数逗号分隔）
   *
   * @param ids 菜单ID字符串，多个以英文逗号(,)分割
   */
  static deleteByIds(ids: string) {
    return request({
      url: "/api/v1/menus/" + ids,
      method: "delete",
    });
  }
}

export default MenuAPI;
