import { OptionType } from "@/types";
import request from "@/utils/request";
import {
  Algorithm,
  AlgorithmAuditForm,
  AlgorithmMonitorVO,
  AlgorithmQuery,
  AlgorithmVersionForm,
  AlgorithmVersionVO,
} from "./model";

class AlgorithmAPI {
  /** 算法树形表格 */
  static getList(queryParams?: AlgorithmQuery) {
    return request<Algorithm[]>({
      url: "/api/v1/algorithms",
      method: "get",
      params: queryParams,
    });
  }

  /** 获取模型下拉选项列表 */
  static getOption() {
    return request<OptionType[]>({
      url: "/api/v1/algorithms/options",
      method: "get",
    });
  }

  /** 获取算法详情 */
  static getAlgorithmInfoById(id: number) {
    return request<Algorithm>({
      url: "/api/v1/algorithms/" + id,
      method: "get",
    });
  }

  /** 新增算法 */
  static add(data: Partial<Algorithm>) {
    return request({
      url: "/api/v1/algorithms",
      method: "post",
      data: data,
    });
  }

  /** 修改算法 */
  static update(id: number, data: Partial<Algorithm>) {
    return request({
      url: "/api/v1/algorithms/" + id,
      method: "put",
      data: data,
    });
  }

  /** 修改算法状态（6生命周期状态：0-5） */
  static updateStatus(id: number, status: number) {
    return request({
      url: `/api/v1/algorithms/${id}/status`,
      method: "put",
      data: { status },
    });
  }

  /** 审核算法（通过/驳回） */
  static auditAlgorithm(id: number, data: AlgorithmAuditForm) {
    return request({
      url: `/api/v1/algorithms/${id}/audit`,
      method: "put",
      data,
    });
  }

  /** 获取算法版本历史 */
  static getVersions(id: number) {
    return request<AlgorithmVersionVO[]>({
      url: `/api/v1/algorithms/${id}/versions`,
      method: "get",
    });
  }

  /** 新增算法版本 */
  static addVersion(id: number, data: AlgorithmVersionForm) {
    return request({
      url: `/api/v1/algorithms/${id}/version`,
      method: "post",
      data,
    });
  }

  /** 版本回滚 */
  static rollbackVersion(id: number, versionId: number) {
    return request({
      url: `/api/v1/algorithms/${id}/rollback`,
      method: "post",
      params: { versionId },
    });
  }

  /** 获取算法监控数据 */
  static getMonitorData(id: number) {
    return request<AlgorithmMonitorVO>({
      url: `/api/v1/algorithms/${id}/monitor`,
      method: "get",
    });
  }

  /** 获取算法统计报表 */
  static getMonitorStats(id: number) {
    return request<AlgorithmMonitorVO>({
      url: `/api/v1/algorithms/${id}/monitor/stats`,
      method: "get",
    });
  }

  /** 删除算法 */
  static deleteByIds(ids: string[]) {
    return request({
      url: "/api/v1/algorithms",
      method: "delete",
      params: { ids: ids.join(",") },
    });
  }
}

export default AlgorithmAPI;
