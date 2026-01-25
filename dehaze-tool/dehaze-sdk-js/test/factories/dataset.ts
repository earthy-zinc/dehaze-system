import {
  DatasetAddForm,
  DatasetUpdateForm,
  DatasetQuery,
  DatasetItemCreateForm,
  DatasetItemUpdateForm,
  DatasetItemQuery,
  ExportTaskRequest,
  BatchDownloadForm,
  ItemFileUpdateForm,
} from "@/api/dataset/model";
import { uniqueName, pageQuery } from "./common";

export function createDatasetForm(overrides: Partial<DatasetAddForm> = {}): DatasetAddForm {
  return {
    parentId: 0,
    name: uniqueName("测试数据集"),
    type: "用户数据集",
    description: "这是一个测试数据集",
    status: "1",
    ...overrides,
  };
}

export function createDatasetUpdateForm(
  overrides: Partial<DatasetUpdateForm> = {}
): DatasetUpdateForm {
  return {
    name: uniqueName("更新后的数据集"),
    ...overrides,
  };
}

export function createDatasetQuery(overrides: Partial<DatasetQuery> = {}): DatasetQuery {
  return pageQuery<DatasetQuery>({
    pageNum: 1,
    pageSize: 10,
    ...overrides,
  });
}

export function createDatasetItemForm(
  datasetId: number,
  overrides: Partial<DatasetItemCreateForm> = {}
): DatasetItemCreateForm {
  return {
    datasetId,
    name: uniqueName("测试数据项"),
    sceneType: "urban",
    ...overrides,
  };
}

export function createDatasetItemUpdateForm(
  overrides: Partial<DatasetItemUpdateForm> = {}
): DatasetItemUpdateForm {
  return {
    name: uniqueName("更新后的数据项"),
    ...overrides,
  };
}

export function createDatasetItemQuery(
  overrides: Partial<DatasetItemQuery> = {}
): DatasetItemQuery {
  return pageQuery<DatasetItemQuery>({
    pageNum: 1,
    pageSize: 10,
    ...overrides,
  });
}

export function createExportTaskRequest(
  overrides: Partial<ExportTaskRequest> = {}
): ExportTaskRequest {
  return {
    includeTypes: ["clear", "hazy"],
    structure: "by_item",
    ...overrides,
  };
}

export function createBatchDownloadForm(
  itemFileIds: number[],
  overrides: Partial<BatchDownloadForm> = {}
): BatchDownloadForm {
  return {
    itemFileIds,
    organizeByItem: true,
    ...overrides,
  };
}

export function createItemFileUpdateForm(
  overrides: Partial<ItemFileUpdateForm> = {}
): ItemFileUpdateForm {
  return {
    description: "测试更新描述",
    ...overrides,
  };
}
