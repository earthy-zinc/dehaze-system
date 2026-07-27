import { OptionType } from "@/types";
import request from "@/utils/request";
import { DeptForm, DeptQuery, DeptVO } from "./model";

class DeptAPI {
  /**
   * 部门树形表格
   *
   * @param queryParams
   */
  static getList(queryParams?: DeptQuery) {
    return request<DeptVO[]>({
      url: "/api/v1/depts",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 部门下拉列表
   */
  static getOptions() {
    return request<OptionType[]>({
      url: "/api/v1/depts/options",
      method: "get",
    });
  }

  /**
   * 获取部门详情
   *
   * @param id 部门ID
   */
  static getFormData(id: number) {
    return request<DeptForm>({
      url: "/api/v1/depts/" + id + "/form",
      method: "get",
    });
  }

  /**
   * 新增部门
   *
   * @param data
   */
  static add(data: DeptForm) {
    return request({
      url: "/api/v1/depts",
      method: "post",
      data: data,
    });
  }

  /**
   * 修改部门
   *
   * @param id 部门ID
   * @param data
   */
  static update(id: number, data: DeptForm) {
    return request({
      url: "/api/v1/depts/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除部门（支持批量，路径参数逗号分隔）
   *
   * @param ids 部门ID字符串，多个以英文逗号(,)分割
   */
  static deleteByIds(ids: string) {
    return request({
      url: "/api/v1/depts/" + ids,
      method: "delete",
    });
  }
}

export default DeptAPI;
