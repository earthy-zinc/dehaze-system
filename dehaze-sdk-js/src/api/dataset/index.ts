import {
  BatchDeleteForm,
  BatchDeleteResultVO,
  BatchOperationResultVO,
  BatchUploadResultVO,
  Dataset,
  DatasetAddForm,
  DatasetItemCreateForm,
  DatasetItemQuery,
  DatasetItemUpdateForm,
  DatasetItemVO,
  DatasetQuery,
  DatasetUpdateForm,
  ImageUrlVO,
  ItemFileUpdateForm,
} from "./model";
import { OptionType, PageResult } from "@/types";
import request from "@/utils/request";

/**
 * 数据集 API
 */
class DatasetAPI {
  // ==================== 数据集接口 ====================

  /**
   * 分页查询数据集列表
   * @param queryParams 查询参数
   */
  static getList(queryParams?: DatasetQuery) {
    return request<PageResult<Dataset[]>>({
      url: "/api/v1/datasets",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 获取子数据集列表（懒加载）
   * @param parentId 父数据集ID
   */
  static getChildren(parentId: number) {
    return request<Dataset[]>({
      url: `/api/v1/datasets/children/${parentId}`,
      method: "get",
    });
  }

  /**
   * 获取数据集下拉选项列表
   */
  static getOptions() {
    return request<OptionType[]>({
      url: "/api/v1/datasets/options",
      method: "get",
    });
  }

  /**
   * 根据ID获取数据集详细信息
   * @param id 数据集ID
   */
  static getDatasetInfoById(id: number) {
    return request<Dataset>({
      url: `/api/v1/datasets/${id}`,
      method: "get",
    });
  }

  /**
   * 新增数据集
   * @param data 数据集创建表单
   */
  static add(data: DatasetAddForm) {
    return request<number>({
      url: "/api/v1/datasets",
      method: "post",
      data: data,
    });
  }

  /**
   * 修改数据集信息
   * @param id 数据集ID
   * @param data 数据集更新表单
   */
  static update(id: number, data: DatasetUpdateForm) {
    return request<Dataset>({
      url: `/api/v1/datasets/${id}`,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除单个数据集
   * @param id 数据集ID
   */
  static deleteById(id: number) {
    return request({
      url: `/api/v1/datasets/${id}`,
      method: "delete",
    });
  }

  /**
   * 批量删除数据集
   * @param data 批量删除表单
   */
  static batchDelete(data: BatchDeleteForm) {
    return request<BatchDeleteResultVO>({
      url: "/api/v1/datasets/batch",
      method: "delete",
      data: data,
    });
  }
}

/**
 * 数据项 API
 */
class DatasetItemAPI {
  /**
   * 分页查询数据项列表
   * @param queryParams 查询参数
   */
  static getList(queryParams?: DatasetItemQuery) {
    return request<PageResult<DatasetItemVO[]>>({
      url: "/api/v1/dataset-items",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 创建空数据项
   * @param data 数据项创建表单
   */
  static add(data: DatasetItemCreateForm) {
    return request<DatasetItemVO>({
      url: "/api/v1/dataset-items",
      method: "post",
      data: data,
    });
  }

  /**
   * 获取数据项详情
   * @param id 数据项ID
   */
  static getById(id: number) {
    return request<DatasetItemVO>({
      url: `/api/v1/dataset-items/${id}`,
      method: "get",
    });
  }

  /**
   * 修改数据项信息
   * @param id 数据项ID
   * @param data 数据项更新表单
   */
  static update(id: number, data: DatasetItemUpdateForm) {
    return request<DatasetItemVO>({
      url: `/api/v1/dataset-items/${id}`,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除数据项
   * @param id 数据项ID
   */
  static deleteById(id: number) {
    return request({
      url: `/api/v1/dataset-items/${id}`,
      method: "delete",
    });
  }

  /**
   * 创建数据项并上传配对图片
   * @param data 上传表单（使用FormData）
   */
  static uploadImagePair(data: FormData) {
    return request<DatasetItemVO>({
      url: "/api/v1/dataset-items/upload",
      method: "post",
      data: data,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * 批量创建数据项并上传图片
   * @param data 批量上传表单（使用FormData）
   */
  static batchUpload(data: FormData) {
    return request<BatchUploadResultVO>({
      url: "/api/v1/dataset-items/batch",
      method: "post",
      data: data,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * 批量删除数据项
   * @param data 批量删除表单
   */
  static batchDelete(data: BatchDeleteForm) {
    return request<BatchOperationResultVO>({
      url: "/api/v1/dataset-items/batch",
      method: "delete",
      data: data,
    });
  }
}

/**
 * 图片文件 API
 */
class ItemFileAPI {
  /**
   * 上传数据项图片
   * @param data 上传表单（使用FormData）
   */
  static upload(data: FormData) {
    return request<ImageUrlVO>({
      url: "/api/v1/item-files",
      method: "post",
      data: data,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * 获取图片详细信息
   * @param id 图片ID
   */
  static getById(id: number) {
    return request<ImageUrlVO>({
      url: `/api/v1/item-files/${id}`,
      method: "get",
    });
  }

  /**
   * 修改图片信息
   * @param id 图片ID
   * @param data 图片更新表单
   */
  static update(id: number, data: ItemFileUpdateForm) {
    return request({
      url: `/api/v1/item-files/${id}`,
      method: "put",
      data: data,
    });
  }

  /**
   * 删除图片
   * @param id 图片ID
   */
  static deleteById(id: number) {
    return request({
      url: `/api/v1/item-files/${id}`,
      method: "delete",
    });
  }

  /**
   * 批量删除图片
   * @param data 批量删除表单
   */
  static batchDelete(data: BatchDeleteForm) {
    return request<BatchDeleteResultVO>({
      url: "/api/v1/item-files/batch",
      method: "delete",
      data: data,
    });
  }
}

// 导出
export default DatasetAPI;
export { DatasetAPI, DatasetItemAPI, ItemFileAPI };
