/**
 * 数据集模块 API 封装
 *
 * 直接调用 SDK 的 DatasetAPI / DatasetItemAPI / ItemFileAPI，
 * SDK 内部已通过 configAxios 注入 baseURL 和 token，
 * 并已在响应拦截器中剥离 R<T> 包装（直接返回 data 字段）。
 *
 * 注意：导出/下载任务已迁移至统一任务接口 /api/v1/tasks，
 * 请使用 taskApi.create({ type: 'dataset_export' | 'item_download' | 'batch_download', ... }) 创建任务。
 */
import {
  DatasetAPI,
  DatasetItemAPI,
  ItemFileAPI,
} from 'dehaze-sdk-js';
import type {
  Dataset,
  DatasetItem,
  DatasetImage,
  DatasetQuery,
  DatasetItemQuery,
} from '../types/dataset';
import type { PageResult } from 'dehaze-sdk-js';

/** 分页结果 */
export type DatasetPage = PageResult<Dataset[]>;
export type DatasetItemPage = PageResult<DatasetItem[]>;

export const datasetApi = {
  // ==================== 数据集 ====================

  /** 分页查询数据集列表 */
  fetchDatasets(query?: DatasetQuery): Promise<DatasetPage> {
    return DatasetAPI.getList(query);
  },

  /** 懒加载子数据集 */
  fetchChildren(parentId: number): Promise<Dataset[]> {
    return DatasetAPI.getChildren(parentId);
  },

  /** 获取数据集下拉选项 */
  fetchOptions() {
    return DatasetAPI.getOptions();
  },

  /** 获取数据集详情 */
  fetchDatasetDetail(id: number): Promise<Dataset> {
    return DatasetAPI.getDatasetInfoById(id);
  },

  /** 删除数据集 */
  deleteDataset(id: number) {
    return DatasetAPI.deleteById(id);
  },

  /** 批量删除数据集 */
  batchDeleteDatasets(ids: number[]) {
    return DatasetAPI.batchDelete({ ids });
  },

  // ==================== 数据项 ====================

  /** 分页查询数据项列表 */
  fetchDatasetItems(query?: DatasetItemQuery): Promise<DatasetItemPage> {
    return DatasetItemAPI.getList(query);
  },

  /** 获取数据项详情（含配对图片） */
  fetchItemDetail(id: number): Promise<DatasetItem> {
    return DatasetItemAPI.getById(id);
  },

  /** 删除数据项 */
  deleteItem(id: number) {
    return DatasetItemAPI.deleteById(id);
  },

  /** 批量删除数据项 */
  batchDeleteItems(ids: number[]) {
    return DatasetItemAPI.batchDelete({ ids });
  },

  // ==================== 图片文件 ====================

  /** 获取图片详情 */
  fetchImageDetail(id: number): Promise<DatasetImage> {
    return ItemFileAPI.getById(id);
  },

  /** 删除图片 */
  deleteImage(id: number) {
    return ItemFileAPI.deleteById(id);
  },

  /** 批量删除图片 */
  batchDeleteImages(ids: number[]) {
    return ItemFileAPI.batchDelete({ ids });
  },
};
