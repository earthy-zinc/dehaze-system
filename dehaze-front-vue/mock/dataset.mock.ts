import { defineMock } from "./base";
import datasetVO from "./data/DatasetVO";
export default defineMock([
  {
    url: "dataset",
    method: ["GET"],
    body: (request) => {
      return {
        code: "00000",
        data: datasetVO,
        msg: "一切ok",
      };
    },
  },
  {
    url: "dataset/:id",
    method: ["PUT"],
    body({ body }) {
      return {
        code: "00000",
        data: null,
        msg: "修改数据集" + body.name + "成功",
      };
    },
  },
  {
    url: "dataset/:id/images",
    method: ["GET"],
    body: {
      code: "00000",
      data: [],
      msg: "一切ok",
    },
  },
]);
