import { create } from "zustand";
import Taro from "@tarojs/taro";
import { DatasetAPI, DatasetItemAPI } from "dehaze-sdk-js";
import type {
  Dataset,
  DatasetItemVO,
  ImageUrlVO,
  DatasetAddForm,
  DatasetUpdateForm,
  OptionType,
} from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";
import { isImageAnnotated, AnnotationFilter } from "../services/imageUtils";

// 状态类型定义
interface DatasetState {
  // 视图状态
  currentView: "list" | "detail";
  currentDatasetId: number | null;

  // 数据集列表（根级）
  datasets: Dataset[];
  datasetsLoading: boolean;
  datasetsError: string | null;
  datasetsPage: number;
  datasetsHasMore: boolean;
  datasetsTotal: number;

  // 树形结构：展开的节点 ID 和子节点缓存
  expandedIds: number[];
  childrenMap: Record<number, Dataset[]>;
  childrenLoading: Record<number, boolean>;

  // 数据集下拉选项（用于新增/编辑表单的父级选择）
  datasetOptions: OptionType[];

  // 当前数据集详情
  currentDataset: Dataset | null;
  datasetDetailLoading: boolean;
  datasetDetailError: string | null;

  // 图片列表（由数据项展平而来）
  images: ImageUrlVO[];
  imagesLoading: boolean;
  imagesError: string | null;
  imagesPage: number;
  imagesHasMore: boolean;
  imagesTotal: number;
  currentAnnotationFilter: AnnotationFilter;

  // 搜索状态
  searchKeyword: string;
  imageSearchKeyword: string;

  // 选中的图片（用于查看器）
  selectedImage: ImageUrlVO | null;
}

interface DatasetStore extends DatasetState {
  setView: (view: "list" | "detail") => void;
  setCurrentDatasetId: (id: number | null) => void;
  fetchDatasets: (
    page?: number,
    search?: string,
    append?: boolean
  ) => Promise<void>;
  fetchChildren: (parentId: number) => Promise<void>;
  toggleExpand: (id: number) => void;
  fetchDatasetOptions: () => Promise<void>;
  createDataset: (data: DatasetAddForm) => Promise<boolean>;
  updateDataset: (id: number, data: DatasetUpdateForm) => Promise<boolean>;
  deleteDataset: (id: number) => Promise<boolean>;
  fetchDatasetDetail: (datasetId: number) => Promise<void>;
  fetchImages: (
    datasetId: number,
    page?: number,
    annotationFilter?: AnnotationFilter,
    search?: string,
    append?: boolean
  ) => Promise<void>;
  setSearchKeyword: (keyword: string) => void;
  setImageSearchKeyword: (keyword: string) => void;
  setAnnotationFilter: (filter: AnnotationFilter) => void;
  setSelectedImage: (image: ImageUrlVO | null) => void;
  resetImages: () => void;
  resetState: () => void;
}

// 初始状态
const initialState: DatasetState = {
  currentView: "list",
  currentDatasetId: null,
  datasets: [],
  datasetsLoading: false,
  datasetsError: null,
  datasetsPage: 1,
  datasetsHasMore: true,
  datasetsTotal: 0,
  expandedIds: [],
  childrenMap: {},
  childrenLoading: {},
  datasetOptions: [],
  currentDataset: null,
  datasetDetailLoading: false,
  datasetDetailError: null,
  images: [],
  imagesLoading: false,
  imagesError: null,
  imagesPage: 1,
  imagesHasMore: true,
  imagesTotal: 0,
  currentAnnotationFilter: "annotated",
  searchKeyword: "",
  imageSearchKeyword: "",
  selectedImage: null,
};

/**
 * 将数据项列表展平为图片列表，并按标注状态过滤：
 * - annotated：仅保留 hazeLevel 非空的图片
 * - unannotated：仅保留 hazeLevel 为空的图片
 */
function flattenItems(
  items: DatasetItemVO[],
  filter: AnnotationFilter
): ImageUrlVO[] {
  const result: ImageUrlVO[] = [];
  items.forEach((item) => {
    if (item.clearImage) {
      const img = item.clearImage;
      if (filter === "annotated" && isImageAnnotated(img.hazeLevel))
        result.push(img);
      else if (filter === "unannotated" && !isImageAnnotated(img.hazeLevel))
        result.push(img);
    }
    if (item.hazyImages) {
      item.hazyImages.forEach((img) => {
        if (filter === "annotated" && isImageAnnotated(img.hazeLevel))
          result.push(img);
        else if (filter === "unannotated" && !isImageAnnotated(img.hazeLevel))
          result.push(img);
      });
    }
  });
  return result;
}

export const useDatasetStore = create<DatasetStore>()((set, get) => ({
  ...initialState,

  setView: (view) => set({ currentView: view }),

  setCurrentDatasetId: (id) => set({ currentDatasetId: id }),

  // 获取数据集列表（根级）
  fetchDatasets: async (page = 1, search = "", append = false) => {
    try {
      set({ datasetsLoading: true, datasetsError: null });

      const size = 10;
      const response = await DatasetAPI.getList({
        keyword: search || undefined,
        pageNum: page,
        pageSize: size,
      });

      const list = response.list || [];
      const total = response.total || 0;
      const hasMore =
        (append ? get().datasets.length + list.length : list.length) < total;

      if (append) {
        set({
          datasets: [...get().datasets, ...list],
          datasetsPage: page,
          datasetsHasMore: hasMore,
          datasetsLoading: false,
        });
      } else {
        set({
          datasets: list,
          datasetsPage: page,
          datasetsTotal: total,
          datasetsHasMore: hasMore,
          datasetsLoading: false,
          datasetsError: null,
        });
      }
    } catch (error: unknown) {
      set({
        datasetsError: getErrorMessage(error, "获取数据集列表失败"),
      });
    }
  },

  // 获取子数据集（懒加载）
  fetchChildren: async (parentId) => {
    try {
      set({
        childrenLoading: { ...get().childrenLoading, [parentId]: true },
      });
      const children = await DatasetAPI.getChildren(parentId);
      set({
        childrenMap: { ...get().childrenMap, [parentId]: children || [] },
        childrenLoading: { ...get().childrenLoading, [parentId]: false },
      });
    } catch (error) {
      console.error("获取子数据集失败:", error);
      set({
        childrenMap: { ...get().childrenMap, [parentId]: [] },
        childrenLoading: { ...get().childrenLoading, [parentId]: false },
      });
    }
  },

  // 切换展开/收起
  toggleExpand: (id) => {
    const isExpanded = get().expandedIds.includes(id);
    set({
      expandedIds: isExpanded
        ? get().expandedIds.filter((i) => i !== id)
        : [...get().expandedIds, id],
    });
    // 展开时若未加载过子节点，触发懒加载
    if (!isExpanded && !get().childrenMap[id]) {
      get().fetchChildren(id);
    }
  },

  // 获取数据集下拉选项（用于父级选择）
  fetchDatasetOptions: async () => {
    try {
      const options = await DatasetAPI.getOptions();
      set({ datasetOptions: options || [] });
    } catch (error) {
      console.error("获取数据集选项失败:", error);
    }
  },

  // 新增数据集
  createDataset: async (data) => {
    try {
      await DatasetAPI.add(data);
      Taro.showToast({ title: "新增成功", icon: "success" });
      // 刷新列表
      get().fetchDatasets(1, get().searchKeyword, false);
      // 刷新选项
      get().fetchDatasetOptions();
      return true;
    } catch (error: unknown) {
      Taro.showToast({
        title: getErrorMessage(error, "新增失败"),
        icon: "none",
      });
      return false;
    }
  },

  // 修改数据集
  updateDataset: async (id, data) => {
    try {
      await DatasetAPI.update(id, data);
      Taro.showToast({ title: "修改成功", icon: "success" });
      // 刷新列表
      get().fetchDatasets(1, get().searchKeyword, false);
      // 刷新选项
      get().fetchDatasetOptions();
      return true;
    } catch (error: unknown) {
      Taro.showToast({
        title: getErrorMessage(error, "修改失败"),
        icon: "none",
      });
      return false;
    }
  },

  // 删除数据集
  deleteDataset: async (id) => {
    try {
      await DatasetAPI.deleteById(id);
      Taro.showToast({ title: "删除成功", icon: "success" });
      // 刷新列表
      get().fetchDatasets(1, get().searchKeyword, false);
      // 刷新选项
      get().fetchDatasetOptions();
      return true;
    } catch (error: unknown) {
      Taro.showToast({
        title: getErrorMessage(error, "删除失败"),
        icon: "none",
      });
      return false;
    }
  },

  // 获取数据集详情
  fetchDatasetDetail: async (datasetId) => {
    try {
      set({ datasetDetailLoading: true, datasetDetailError: null });

      const dataset = await DatasetAPI.getDatasetInfoById(datasetId);
      set({
        currentDataset: dataset,
        datasetDetailLoading: false,
        datasetDetailError: null,
      });
    } catch (error: unknown) {
      set({
        datasetDetailError: getErrorMessage(error, "获取数据集详情失败"),
      });
    }
  },

  // 获取图片列表（通过数据项接口获取后展平，按标注状态过滤）
  fetchImages: async (
    datasetId,
    page = 1,
    annotationFilter = "annotated",
    search = "",
    append = false
  ) => {
    try {
      set({ imagesLoading: true, imagesError: null });

      const size = 20;
      const response = await DatasetItemAPI.getList({
        datasetId,
        keyword: search || undefined,
        pageNum: page,
        pageSize: size,
      });

      const items = response.list || [];
      const flattened = flattenItems(items, annotationFilter);
      const total = response.total || 0;
      const hasMore = items.length === size;

      if (append) {
        set({
          images: [...get().images, ...flattened],
          imagesPage: page,
          imagesHasMore: hasMore,
          imagesLoading: false,
        });
      } else {
        set({
          images: flattened,
          imagesPage: page,
          imagesTotal: total,
          imagesHasMore: hasMore,
          imagesLoading: false,
          imagesError: null,
        });
      }
    } catch (error: unknown) {
      set({
        imagesError: getErrorMessage(error, "获取图片列表失败"),
      });
    }
  },

  // 其他简化 actions
  setSearchKeyword: (keyword) => set({ searchKeyword: keyword }),
  setImageSearchKeyword: (keyword) => set({ imageSearchKeyword: keyword }),
  setAnnotationFilter: (filter) => set({ currentAnnotationFilter: filter }),
  setSelectedImage: (image) => set({ selectedImage: image }),

  resetImages: () =>
    set({
      images: [],
      imagesPage: 1,
      imagesHasMore: true,
      imagesTotal: 0,
      imageSearchKeyword: "",
      currentAnnotationFilter: "annotated",
    }),

  resetState: () => set({ ...initialState }),
}));

/**
 * 兼容 hook：订阅 zustand store，返回 { state, ...actions }，
 * 与原 Context + Reducer 的 useDataset 返回结构保持一致。
 */
export function useDataset() {
  const state = useDatasetStore();
  const actions = useDatasetStore.getState();
  return {
    state,
    // Actions
    setView: actions.setView,
    setCurrentDatasetId: actions.setCurrentDatasetId,
    fetchDatasets: actions.fetchDatasets,
    fetchChildren: actions.fetchChildren,
    toggleExpand: actions.toggleExpand,
    fetchDatasetOptions: actions.fetchDatasetOptions,
    createDataset: actions.createDataset,
    updateDataset: actions.updateDataset,
    deleteDataset: actions.deleteDataset,
    fetchDatasetDetail: actions.fetchDatasetDetail,
    fetchImages: actions.fetchImages,
    setSearchKeyword: actions.setSearchKeyword,
    setImageSearchKeyword: actions.setImageSearchKeyword,
    setAnnotationFilter: actions.setAnnotationFilter,
    setSelectedImage: actions.setSelectedImage,
    resetImages: actions.resetImages,
    resetState: actions.resetState,
  };
}