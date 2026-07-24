import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  EvalLogQuery,
  EvalLogVO,
  EvaluationForm,
  EvaluationResultVO,
  PredLogQuery,
  PredLogVO,
  PredictionForm,
  PredictionResultVO,
} from "./model";

class ModelAPI {
  /** 执行模型预测（去雾处理） */
  static predict(data: PredictionForm) {
    return request<any, PredictionResultVO>({
      url: "/api/v1/prediction",
      method: "post",
      data,
    });
  }

  /** 查询预测任务状态 */
  static getPredTaskStatus(taskId: number) {
    return request<any, PredictionResultVO>({
      url: `/api/v1/prediction/${taskId}`,
      method: "get",
    });
  }

  /** 获取预测日志分页列表 */
  static getPredLogs(query?: PredLogQuery) {
    return request<any, PageResult<PredLogVO[]>>({
      url: "/api/v1/prediction/logs",
      method: "get",
      params: query,
    });
  }

  /** 执行效果评估（PSNR/SSIM/LPIPS等） */
  static evaluate(data: EvaluationForm) {
    return request<any, EvaluationResultVO>({
      url: "/api/v1/evaluation",
      method: "post",
      data,
    });
  }

  /** 查询评估任务状态 */
  static getEvalTaskStatus(taskId: number) {
    return request<any, EvaluationResultVO>({
      url: `/api/v1/evaluation/${taskId}`,
      method: "get",
    });
  }

  /** 获取评估日志分页列表 */
  static getEvalLogs(query?: EvalLogQuery) {
    return request<any, PageResult<EvalLogVO[]>>({
      url: "/api/v1/evaluation/logs",
      method: "get",
      params: query,
    });
  }
}

export default ModelAPI;
