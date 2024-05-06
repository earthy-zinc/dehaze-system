import { DatasetItem, DatasetQuery, DatasetVO } from "@/api/dataset/model";
import request from "@/utils/request";

class DatasetAPI {
  /**
   * 数据集树形表格
   * @param queryParams
   */
  static getList(queryParams?: DatasetQuery) {
    return request<any, DatasetVO[]>({
      url: "/api/v1/dataset",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 数据集下拉列表
   */
  static getOptions() {
    return request<any, OptionType[]>({
      url: "/api/v1/dataset/options",
      method: "get",
    });
  }

  /**
   * 获取数据集详细图片
   *
   * @param id
   */
  static getFormData(id: number) {
    return request<any, DatasetItem[]>({
      url: "/api/v1/dataset/" + id + "/form",
      method: "get",
    });
  }

  /**
   * 新增数据集
   *
   * @param data
   */
  static add(data: DatasetVO) {
    return request({
      url: "/api/v1/dataset",
      method: "post",
      data: data,
    });
  }

  /**
   *  修改部门
   *
   * @param id
   * @param data
   */
  static update(id: number, data: DatasetVO) {
    return request({
      url: "/api/v1/dataset/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除数据集
   *
   * @param ids
   */
  static deleteByIds(ids: string) {
    return request({
      url: "/api/v1/dataset/" + ids,
      method: "delete",
    });
  }
}

export default DatasetAPI;
