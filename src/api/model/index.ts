import request from "@/utils/requestPy";
import { EvalParam, EvalResult, PredParam, PredResult } from "./model";

class ModelAPI {
  static prediction(params: PredParam) {
    return request<any, PredResult[]>({
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
