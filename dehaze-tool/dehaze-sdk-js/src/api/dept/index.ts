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
    return request<any, DeptVO[]>({
      url: "/api/v1/depts",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 部门下拉列表
   */
  static getOptions() {
    return request<any, OptionType[]>({
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
    return request<any, DeptForm>({
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
   * 删除单个部门
   *
   * @param id 部门ID
   */
  static deleteById(id: number) {
    return request({
      url: "/api/v1/depts/" + id,
      method: "delete",
    });
  }

  /**
   * 批量删除部门（RequestBody JSON）
   *
   * @param ids 部门ID数组
   */
  static batchDelete(ids: number[]) {
    return request({
      url: "/api/v1/depts/batch",
      method: "delete",
      data: { ids },
    });
  }
}

export default DeptAPI;
