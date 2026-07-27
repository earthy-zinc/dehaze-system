import { PageResult } from "@/types";
import request from "@/utils/request";
import { HistoryForm, HistoryQuery, HistoryUpdateForm, InputHistoryVO } from "./model";

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

  /** 更新历史记录（如添加收藏） */
  static update(id: number, data: HistoryUpdateForm) {
    return request({
      url: `/api/v1/image-input/history/${id}`,
      method: "put",
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
  static clearAll() {
    return request<number>({
      url: "/api/v1/image-input/history/clear",
      method: "delete",
    });
  }

  /** 同步本地与云端历史记录 */
  static sync() {
    return request<number>({
      url: "/api/v1/image-input/history/sync",
      method: "post",
    });
  }
}

export default ImageInputHistoryAPI;
