import { PageResult } from "@/types";
import request from "@/utils/request";
import { HistoryForm, HistoryQuery, InputHistoryVO } from "./model";

class ImageInputHistoryAPI {
  /** 分页查询历史记录 */
  static getPage(query?: HistoryQuery) {
    return request<PageResult<InputHistoryVO[]>>({
      url: "/api/v1/image-input/history",
      method: "get",
      params: query,
    });
  }

  /** 获取历史记录详情 */
  static getById(id: number) {
    return request<InputHistoryVO>({
      url: `/api/v1/image-input/history/${id}`,
      method: "get",
    });
  }

  /** 创建历史记录 */
  static create(data: HistoryForm) {
    return request<number>({
      url: "/api/v1/image-input/history",
      method: "post",
      data,
    });
  }

  /** 删除单条历史记录 */
  static deleteById(id: number) {
    return request({
      url: `/api/v1/image-input/history/${id}`,
      method: "delete",
    });
  }

  /** 批量删除历史记录 */
  static batchDelete(ids: number[]) {
    return request<number>({
      url: "/api/v1/image-input/history/batch",
      method: "delete",
      data: { ids },
    });
  }

  /** 清空所有历史记录 */
  static clearAll(confirm: boolean) {
    return request<number>({
      url: "/api/v1/image-input/history/clear",
      method: "delete",
      params: { confirm },
    });
  }
}

export default ImageInputHistoryAPI;
