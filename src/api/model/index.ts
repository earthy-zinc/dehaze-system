import request from "@/utils/request";
import { EvalParam, EvalResult, PredParam } from "./model";

class ModelAPI {
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
