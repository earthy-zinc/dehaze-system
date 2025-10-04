import { defineMock } from "./base";
import modelVO from "./data/ModelVO";
import modelOptionsVO from "./data/ModelOptionsVO";
export default defineMock([
  {
    url: "algorithm",
    method: ["GET"],
    body: (request) => {
      return {
        code: "00000",
        data: modelVO,
        msg: "一切ok",
      };
    },
  },
  {
    url: "algorithm",
    method: ["POST"],
    body: (request) => {
      return {
        code: "00000",
        data: null,
        msg: "一切ok",
      };
    },
  },
  {
    url: "algorithm/options",
    method: ["GET"],
    body: (request) => {
      return {
        code: "00000",
        data: modelOptionsVO,
        msg: "一切ok",
      };
    },
  },
  {
    url: "algorithm/:id",
    method: ["PUT"],
    body({ body }) {
      return {
        code: "00000",
        data: null,
        msg: "修改模型算法" + body.name + "成功",
      };
    },
  },
  {
    url: "algorithm/:id/images",
    method: ["GET"],
    body: {
      code: "00000",
      data: [],
      msg: "一切ok",
    },
  },
]);
