import { OptionType } from "@/types";
import request from "@/utils/request";
import { PredictionResultVO } from "@/api/model/model";
import {
  Algorithm,
  AlgorithmAuditForm,
  AlgorithmCompareForm,
  AlgorithmCompareVO,
  AlgorithmDetailVO,
  AlgorithmMonitorVO,
  AlgorithmMonitorStatsItemVO,
  AlgorithmQuery,
  AlgorithmRecommendForm,
  AlgorithmRecommendResult,
  AlgorithmSelectNodeVO,
  AlgorithmTestForm,
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

  /** 获取所有算法扁平列表（不分页，不构建树形） */
  static listAll() {
    return request<Algorithm[]>({
      url: "/api/v1/algorithms/list",
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

  /** 新增算法（返回算法ID） */
  static add(data: Partial<Algorithm>) {
    return request<number>({
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

  /** 修改算法状态（状态值 1-6：1草稿/2测试中/3待审核/4已发布/5已停用/6已归档） */
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

  /** 获取算法统计报表（days 统计天数，默认 7） */
  static getMonitorStats(id: number, days?: number) {
    return request<AlgorithmMonitorStatsItemVO[]>({
      url: `/api/v1/algorithms/${id}/monitor/stats`,
      method: "get",
      params: days !== undefined ? { days } : undefined,
    });
  }

  /** 算法对比（最多 3 个） */
  static compare(data: AlgorithmCompareForm) {
    return request<AlgorithmCompareVO[]>({
      url: "/api/v1/algorithms/select/compare",
      method: "post",
      data,
    });
  }

  /** 获取算法选择树（仅已发布算法） */
  static tree(taskType?: string) {
    return request<AlgorithmSelectNodeVO[]>({
      url: "/api/v1/algorithms/select/tree",
      method: "get",
      params: taskType ? { taskType } : undefined,
    });
  }

  /** 获取算法详情（含样例效果图、评分、使用次数） */
  static getSelectDetail(id: number) {
    return request<AlgorithmDetailVO>({
      url: "/api/v1/algorithms/select/" + id,
      method: "get",
    });
  }

  /** 上传自定义图片测试算法效果 */
  static test(id: number, data: AlgorithmTestForm) {
    return request<PredictionResultVO>({
      url: `/api/v1/algorithms/select/${id}/test`,
      method: "post",
      data,
    });
  }

  /** 搜索算法（关键词/拼音/标签） */
  static search(keyword: string, taskType?: string) {
    return request<AlgorithmSelectNodeVO[]>({
      url: "/api/v1/algorithms/select/search",
      method: "get",
      params: taskType ? { keyword, taskType } : { keyword },
    });
  }

  /** 算法推荐匹配（基于关键词/任务类型/样例算法） */
  static recommend(data: AlgorithmRecommendForm) {
    return request<AlgorithmRecommendResult>({
      url: "/api/v1/algorithms/select/recommend",
      method: "post",
      data,
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
