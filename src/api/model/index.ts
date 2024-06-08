import request from "@/utils/request";
import { EvalParam, EvalResult, PredParam } from "./model";

class ModelAPI {
  /**
   * 获取模型下拉选项列表
   */
  static getOption() {
    return request<any, OptionType[]>({
      url: "/model/options",
      method: "get",
    });
  }

  static prediction(params: PredParam) {
    return request({
      url: "/model/prediction",
      method: "get",
      params,
    });
  }

  static evaluation(params: EvalParam) {
    return request<any, EvalResult[]>({
      url: "/model/evaluation",
      method: "get",
      params,
    });
  }
}

export default ModelAPI;
