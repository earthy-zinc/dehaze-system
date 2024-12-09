import {
  Dataset,
  DatasetQuery,
  ImageFileInfo,
  ImageItem,
  ImageItemQuery,
} from "@/api/dataset/model";
import request from "@/utils/request";

class DatasetAPI {
  /**
   * 数据集树形表格
   * @param queryParams
   */
  static getList(queryParams?: DatasetQuery) {
    return request<any, Dataset[]>({
      url: "/api/v1/dataset",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 获取数据集下拉列表
   */
  static getOptions() {
    return request<any, OptionType[]>({
      url: "/api/v1/dataset/options",
      method: "get",
    });
  }

  /**
   * 根据Id获取数据集信息
   * @param id 数据集id
   */
  static getDatasetInfoById(id: number) {
    return request<any, Dataset>({
      url: "/api/v1/dataset/" + id,
      method: "get",
    });
  }

  /**
   * 获取数据集详细图片
   *
   * @param id
   * @param queryParams
   */
  static getImageItem(id: number, queryParams: ImageItemQuery) {
    return request<any, PageResult<ImageItem[]>>({
      url: "/api/v1/dataset/" + id + "/images",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 新增数据集
   *
   * @param data
   */
  static add(data: Dataset) {
    return request({
      url: "/api/v1/dataset",
      method: "post",
      data: data,
    });
  }

  /**
   *  修改数据集
   *
   * @param id
   * @param data
   */
  static update(id: number, data: Dataset) {
    return request({
      url: "/api/v1/dataset/" + id,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除数据集
   *
   * @param ids
   */
  static deleteByIds(ids: string[]) {
    return request({
      url: "/api/v1/dataset/" + ids,
      method: "delete",
    });
  }

  /**
   * 新增数据项
   * @param datasetId
   * @param name
   */
  static addDatasetItem(datasetId: number, name?: string) {
    return request<any, number>({
      url: "/api/v1/dataset/item",
      method: "post",
      params: {
        datasetId,
        name,
      },
    });
  }

  static updateDatasetItem(datasetItemId: number, name: string) {
    return request({
      url: "/api/v1/dataset/item/",
      method: "put",
      params: {
        datasetItemId,
        name,
      },
    });
  }

  static deleteDatasetItem(datasetItemId: number) {
    return request({
      url: "/api/v1/dataset/item",
      method: "delete",
      params: {
        datasetItemId,
      },
    });
  }

  static uploadItemImage(
    datasetId: number,
    datasetItemId: number,
    type: string,
    file: File,
    description?: string
  ) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("datasetId", datasetId.toString());
    formData.append("datasetItemId", datasetItemId.toString());
    formData.append("type", type);
    if (description) {
      formData.append("description", description);
    }
    return request<any, ImageFileInfo>({
      url: "/api/v1/dataset/image",
      method: "post",
      data: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  static updateItemImage(
    itemFileId: number,
    type: string,
    description?: string
  ) {
    return request({
      url: "/api/v1/dataset/image/",
      method: "put",
      params: {
        itemFileId,
        type,
        description,
      },
    });
  }

  static deleteItemImage(itemFileId: number) {
    return request({
      url: "/api/v1/dataset/image",
      method: "delete",
      params: { itemFileId },
    });
  }
}

export default DatasetAPI;
