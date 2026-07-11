import {
  BatchDeleteForm,
  BatchDownloadForm,
  Dataset,
  DatasetAddForm,
  DatasetItemCreateForm,
  DatasetItemQuery,
  DatasetItemUpdateForm,
  DatasetItemVO,
  DatasetQuery,
  DatasetOption,
  DatasetUpdateForm,
  DownloadTaskVO,
  ExportTaskRequest,
  ImageUrlVO,
  ItemFileUpdateForm,
  TaskQuery,
} from "./model";
import { PageResult } from "@/types";
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
    return request<any, PageResult<Dataset[]>>({
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
    return request<any, Dataset[]>({
      url: `/api/v1/datasets/children/${parentId}`,
      method: "get",
    });
  }

  /**
   * 获取数据集下拉选项列表
   */
  static getOptions() {
    return request<any, DatasetOption[]>({
      url: "/api/v1/datasets/options",
      method: "get",
    });
  }

  /**
   * 根据ID获取数据集详细信息
   * @param id 数据集ID
   */
  static getDatasetInfoById(id: number) {
    return request<any, Dataset>({
      url: `/api/v1/datasets/${id}`,
      method: "get",
    });
  }

  /**
   * 新增数据集
   * @param data 数据集创建表单
   */
  static add(data: DatasetAddForm) {
    return request<any, Dataset>({
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
    return request<any, Dataset>({
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
    return request<any, { successIds: number[]; failedItems: { id: number; reason: string }[] }>({
      url: "/api/v1/datasets/batch",
      method: "delete",
      data: data,
    });
  }

  /**
   * 创建数据集导出任务
   * @param id 数据集ID
   * @param data 导出任务请求
   */
  static createExportTask(id: number, data?: ExportTaskRequest) {
    return request<any, DownloadTaskVO>({
      url: `/api/v1/datasets/${id}/export`,
      method: "post",
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
    return request<any, PageResult<DatasetItemVO[]>>({
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
    return request<any, DatasetItemVO>({
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
    return request<any, DatasetItemVO>({
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
    return request<any, DatasetItemVO>({
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
    return request<any, DatasetItemVO>({
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
    return request<
      any,
      {
        total: number;
        succeeded: number;
        failed: number;
        successItems: { id: number; name: string; fileCount: number }[];
        failedItems: { fileName: string; reason: string }[];
      }
    >({
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
    return request<
      any,
      {
        successCount: number;
        failedCount: number;
        message: string;
        successIds?: number[];
        failureDetails?: { identifier?: string; reason: string }[];
      }
    >({
      url: "/api/v1/dataset-items/batch",
      method: "delete",
      data: data,
    });
  }

  /**
   * 批量下载数据项图片
   * @param data 批量下载表单
   */
  static batchDownload(data: BatchDownloadForm) {
    return request<any, DownloadTaskVO>({
      url: "/api/v1/dataset-items/batch/download",
      method: "post",
      data: data,
    });
  }

  /**
   * 创建数据项下载任务
   * @param id 数据项ID
   * @param itemFileIds 需要下载的图片ID列表
   */
  static createDownloadTask(id: number, itemFileIds?: number[]) {
    return request<any, DownloadTaskVO>({
      url: `/api/v1/dataset-items/${id}/download/task`,
      method: "post",
      params: { itemFileId: itemFileIds },
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
    return request<any, ImageUrlVO>({
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
    return request<any, ImageUrlVO>({
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
    return request<
      any,
      {
        successIds: number[];
        failedItems: { id?: number; reason: string }[];
        successCount: number;
        failedCount: number;
      }
    >({
      url: "/api/v1/item-files/batch",
      method: "delete",
      data: data,
    });
  }
}

// 导出旧接口（兼容性）
class LegacyDatasetAPI {
  /**
   * @deprecated 使用 DatasetItemAPI.add 替代
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

  /**
   * @deprecated 使用 DatasetItemAPI.update 替代
   * 修改数据项
   * @param datasetItemId
   * @param name
   */
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

  /**
   * @deprecated 使用 DatasetItemAPI.deleteById 替代
   * 删除数据项
   * @param datasetItemId
   */
  static deleteDatasetItem(datasetItemId: number) {
    return request({
      url: "/api/v1/dataset/item",
      method: "delete",
      params: {
        datasetItemId,
      },
    });
  }

  /**
   * @deprecated 使用 ItemFileAPI.upload 替代
   * 上传图片
   * @param datasetId
   * @param datasetItemId
   * @param type
   * @param file
   * @param description
   */
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
    return request<
      any,
      {
        id: number;
        datasetItemId: number;
        fileId: number;
        type: string;
        description: string;
        url: string;
      }
    >({
      url: "/api/v1/dataset/image",
      method: "post",
      data: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * @deprecated 使用 ItemFileAPI.update 替代
   * 修改图片信息
   * @param itemFileId
   * @param type
   * @param description
   */
  static updateItemImage(itemFileId: number, type: string, description?: string) {
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

  /**
   * @deprecated 使用 ItemFileAPI.deleteById 替代
   * 删除图片
   * @param itemFileId
   */
  static deleteItemImage(itemFileId: number) {
    return request({
      url: "/api/v1/dataset/image",
      method: "delete",
      params: { itemFileId },
    });
  }
}

/**
 * 任务 API
 */
class ExportTaskAPI {
  /**
   * 分页查询任务列表
   * @param queryParams 查询参数
   */
  static getList(queryParams?: TaskQuery) {
    return request<any, PageResult<DownloadTaskVO[]>>({
      url: "/api/v1/tasks",
      method: "get",
      params: queryParams,
    });
  }

  /**
   * 查询任务状态
   * @param taskId 任务ID
   */
  static getTaskStatus(taskId: string) {
    return request<any, DownloadTaskVO>({
      url: `/api/v1/tasks/${taskId}`,
      method: "get",
    });
  }

  /**
   * 取消任务
   * @param taskId 任务ID
   */
  static cancelTask(taskId: string) {
    return request({
      url: `/api/v1/tasks/${taskId}`,
      method: "delete",
    });
  }
}

// 导出
export default DatasetAPI;
export { DatasetAPI, DatasetItemAPI, ItemFileAPI, ExportTaskAPI, LegacyDatasetAPI };
