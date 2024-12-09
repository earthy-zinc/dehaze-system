import request from "@/utils/request";
import { Algorithm, AlgorithmQuery } from "./model";

class AlgorithmAPI {
  /**
   * 算法树形表格
   * @param queryParams
   */
  static getList(queryParams?: AlgorithmQuery) {
    return request<any, Algorithm[]>({
      url: "/api/v1/algorithm",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 获取模型下拉选项列表
   */
  static getOption() {
    return request<any, OptionType[]>({
      url: "/api/v1/algorithm/options",
      method: "get",
    });
  }

  /**
   * 获取算法详情
   */
  static getAlgorithmInfoById(id: number) {
    return request<any, Algorithm>({
      url: "/api/v1/algorithm/" + id,
      method: "get",
    });
  }

  /**
   * 新增算法
   *
   * @param data
   */
  static add(data: Algorithm) {
    return request({
      url: "/api/v1/algorithm",
      method: "post",
      data: data,
    });
  }

  /**
   *  修改算法
   *
   * @param id
   * @param data
   */
  static update(id: number, data: Algorithm) {
    return request({
      url: "/api/v1/algorithm/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除算法
   *
   * @param ids
   */
  static deleteByIds(ids: string[]) {
    return request({
      url: "/api/v1/algorithm/" + ids,
      method: "delete",
    });
  }
}

export default AlgorithmAPI;
