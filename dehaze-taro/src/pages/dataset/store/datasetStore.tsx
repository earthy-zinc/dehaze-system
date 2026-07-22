import {
  createContext,
  useContext,
  useReducer,
  ReactNode,
  useCallback,
} from "react";
import Taro from "@tarojs/taro";
import { DatasetAPI, DatasetItemAPI } from "dehaze-sdk-js";
import type {
  Dataset,
  DatasetItemVO,
  ImageUrlVO,
  DatasetAddForm,
  DatasetUpdateForm,
  DatasetOption,
} from "dehaze-sdk-js";
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
  datasetOptions: DatasetOption[];

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

// 动作类型定义
type DatasetAction =
  | { type: "SET_VIEW"; payload: "list" | "detail" }
  | { type: "SET_CURRENT_DATASET_ID"; payload: number | null }
  | { type: "SET_DATASETS_LOADING"; payload: boolean }
  | { type: "SET_DATASETS_ERROR"; payload: string | null }
  | {
      type: "SET_DATASETS";
      payload: {
        datasets: Dataset[];
        page: number;
        total: number;
        hasMore: boolean;
      };
    }
  | {
      type: "APPEND_DATASETS";
      payload: { datasets: Dataset[]; page: number; hasMore: boolean };
    }
  | { type: "SET_EXPANDED"; payload: number[] }
  | { type: "TOGGLE_EXPAND"; payload: number }
  | { type: "SET_CHILDREN"; payload: { parentId: number; children: Dataset[] } }
  | {
      type: "SET_CHILDREN_LOADING";
      payload: { parentId: number; loading: boolean };
    }
  | { type: "SET_DATASET_OPTIONS"; payload: DatasetOption[] }
  | { type: "SET_DATASET_DETAIL_LOADING"; payload: boolean }
  | { type: "SET_DATASET_DETAIL_ERROR"; payload: string | null }
  | { type: "SET_CURRENT_DATASET"; payload: Dataset | null }
  | { type: "SET_IMAGES_LOADING"; payload: boolean }
  | { type: "SET_IMAGES_ERROR"; payload: string | null }
  | {
      type: "SET_IMAGES";
      payload: {
        images: ImageUrlVO[];
        page: number;
        total: number;
        hasMore: boolean;
      };
    }
  | {
      type: "APPEND_IMAGES";
      payload: { images: ImageUrlVO[]; page: number; hasMore: boolean };
    }
  | { type: "SET_ANNOTATION_FILTER"; payload: AnnotationFilter }
  | { type: "SET_SEARCH_KEYWORD"; payload: string }
  | { type: "SET_IMAGE_SEARCH_KEYWORD"; payload: string }
  | { type: "SET_SELECTED_IMAGE"; payload: ImageUrlVO | null }
  | { type: "RESET_IMAGES" }
  | { type: "RESET_STATE" };

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

// Reducer
function datasetReducer(
  state: DatasetState,
  action: DatasetAction
): DatasetState {
  switch (action.type) {
    case "SET_VIEW":
      return { ...state, currentView: action.payload };
    case "SET_CURRENT_DATASET_ID":
      return { ...state, currentDatasetId: action.payload };
    case "SET_DATASETS_LOADING":
      return { ...state, datasetsLoading: action.payload };
    case "SET_DATASETS_ERROR":
      return { ...state, datasetsError: action.payload };
    case "SET_DATASETS":
      return {
        ...state,
        datasets: action.payload.datasets,
        datasetsPage: action.payload.page,
        datasetsTotal: action.payload.total,
        datasetsHasMore: action.payload.hasMore,
        datasetsLoading: false,
        datasetsError: null,
      };
    case "APPEND_DATASETS":
      return {
        ...state,
        datasets: [...state.datasets, ...action.payload.datasets],
        datasetsPage: action.payload.page,
        datasetsHasMore: action.payload.hasMore,
        datasetsLoading: false,
      };
    case "SET_EXPANDED":
      return { ...state, expandedIds: action.payload };
    case "TOGGLE_EXPAND": {
      const id = action.payload;
      const isExpanded = state.expandedIds.includes(id);
      return {
        ...state,
        expandedIds: isExpanded
          ? state.expandedIds.filter((i) => i !== id)
          : [...state.expandedIds, id],
      };
    }
    case "SET_CHILDREN":
      return {
        ...state,
        childrenMap: {
          ...state.childrenMap,
          [action.payload.parentId]: action.payload.children,
        },
        childrenLoading: {
          ...state.childrenLoading,
          [action.payload.parentId]: false,
        },
      };
    case "SET_CHILDREN_LOADING":
      return {
        ...state,
        childrenLoading: {
          ...state.childrenLoading,
          [action.payload.parentId]: action.payload.loading,
        },
      };
    case "SET_DATASET_OPTIONS":
      return { ...state, datasetOptions: action.payload };
    case "SET_DATASET_DETAIL_LOADING":
      return { ...state, datasetDetailLoading: action.payload };
    case "SET_DATASET_DETAIL_ERROR":
      return { ...state, datasetDetailError: action.payload };
    case "SET_CURRENT_DATASET":
      return {
        ...state,
        currentDataset: action.payload,
        datasetDetailLoading: false,
        datasetDetailError: null,
      };
    case "SET_IMAGES_LOADING":
      return { ...state, imagesLoading: action.payload };
    case "SET_IMAGES_ERROR":
      return { ...state, imagesError: action.payload };
    case "SET_IMAGES":
      return {
        ...state,
        images: action.payload.images,
        imagesPage: action.payload.page,
        imagesTotal: action.payload.total,
        imagesHasMore: action.payload.hasMore,
        imagesLoading: false,
        imagesError: null,
      };
    case "APPEND_IMAGES":
      return {
        ...state,
        images: [...state.images, ...action.payload.images],
        imagesPage: action.payload.page,
        imagesHasMore: action.payload.hasMore,
        imagesLoading: false,
      };
    case "SET_ANNOTATION_FILTER":
      return { ...state, currentAnnotationFilter: action.payload };
    case "SET_SEARCH_KEYWORD":
      return { ...state, searchKeyword: action.payload };
    case "SET_IMAGE_SEARCH_KEYWORD":
      return { ...state, imageSearchKeyword: action.payload };
    case "SET_SELECTED_IMAGE":
      return { ...state, selectedImage: action.payload };
    case "RESET_IMAGES":
      return {
        ...state,
        images: [],
        imagesPage: 1,
        imagesHasMore: true,
        imagesTotal: 0,
        imageSearchKeyword: "",
        currentAnnotationFilter: "annotated",
      };
    case "RESET_STATE":
      return initialState;
    default:
      return state;
  }
}

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

// Context
const DatasetContext = createContext<{
  state: DatasetState;
  dispatch: React.Dispatch<DatasetAction>;
} | null>(null);

// Provider
export function DatasetProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(datasetReducer, initialState);

  return (
    <DatasetContext.Provider value={{ state, dispatch }}>
      {children}
    </DatasetContext.Provider>
  );
}

// Hook
export function useDataset() {
  const context = useContext(DatasetContext);
  if (!context) {
    throw new Error("useDataset must be used within a DatasetProvider");
  }

  const { state, dispatch } = context;

  // Actions
  const setView = useCallback((view: "list" | "detail") => {
    dispatch({ type: "SET_VIEW", payload: view });
  }, []);

  const setCurrentDatasetId = useCallback((id: number | null) => {
    dispatch({ type: "SET_CURRENT_DATASET_ID", payload: id });
  }, []);

  // 获取数据集列表（根级）
  const fetchDatasets = useCallback(
    async (page = 1, search = "", append = false) => {
      try {
        dispatch({ type: "SET_DATASETS_LOADING", payload: true });
        dispatch({ type: "SET_DATASETS_ERROR", payload: null });

        const pageSize = 10;
        const response = await DatasetAPI.getList({
          keyword: search || undefined,
          pageNum: page,
          pageSize,
        });

        const list = (response.list as unknown as Dataset[]) || [];
        const total = response.total || 0;
        const hasMore =
          (append ? state.datasets.length + list.length : list.length) < total;

        if (append) {
          dispatch({
            type: "APPEND_DATASETS",
            payload: { datasets: list, page, hasMore },
          });
        } else {
          dispatch({
            type: "SET_DATASETS",
            payload: { datasets: list, page, total, hasMore },
          });
        }
      } catch (error: any) {
        dispatch({
          type: "SET_DATASETS_ERROR",
          payload: error?.message || "获取数据集列表失败",
        });
      }
    },
    [state.datasets.length]
  );

  // 获取子数据集（懒加载）
  const fetchChildren = useCallback(async (parentId: number) => {
    try {
      dispatch({
        type: "SET_CHILDREN_LOADING",
        payload: { parentId, loading: true },
      });
      const children = await DatasetAPI.getChildren(parentId);
      dispatch({
        type: "SET_CHILDREN",
        payload: { parentId, children: children || [] },
      });
    } catch (error) {
      console.error("获取子数据集失败:", error);
      dispatch({ type: "SET_CHILDREN", payload: { parentId, children: [] } });
    }
  }, []);

  // 切换展开/收起
  const toggleExpand = useCallback(
    (id: number) => {
      const isExpanded = state.expandedIds.includes(id);
      dispatch({ type: "TOGGLE_EXPAND", payload: id });
      // 展开时若未加载过子节点，触发懒加载
      if (!isExpanded && !state.childrenMap[id]) {
        fetchChildren(id);
      }
    },
    [state.expandedIds, state.childrenMap, fetchChildren]
  );

  // 获取数据集下拉选项（用于父级选择）
  const fetchDatasetOptions = useCallback(async () => {
    try {
      const options = await DatasetAPI.getOptions();
      dispatch({ type: "SET_DATASET_OPTIONS", payload: options || [] });
    } catch (error) {
      console.error("获取数据集选项失败:", error);
    }
  }, []);

  // 新增数据集
  const createDataset = useCallback(
    async (data: DatasetAddForm) => {
      try {
        await DatasetAPI.add(data);
        Taro.showToast({ title: "新增成功", icon: "success" });
        // 刷新列表
        fetchDatasets(1, state.searchKeyword, false);
        // 刷新选项
        fetchDatasetOptions();
        return true;
      } catch (error: any) {
        Taro.showToast({ title: error?.message || "新增失败", icon: "none" });
        return false;
      }
    },
    [fetchDatasets, fetchDatasetOptions, state.searchKeyword]
  );

  // 修改数据集
  const updateDataset = useCallback(
    async (id: number, data: DatasetUpdateForm) => {
      try {
        await DatasetAPI.update(id, data);
        Taro.showToast({ title: "修改成功", icon: "success" });
        // 刷新列表
        fetchDatasets(1, state.searchKeyword, false);
        // 刷新选项
        fetchDatasetOptions();
        return true;
      } catch (error: any) {
        Taro.showToast({ title: error?.message || "修改失败", icon: "none" });
        return false;
      }
    },
    [fetchDatasets, fetchDatasetOptions, state.searchKeyword]
  );

  // 删除数据集
  const deleteDataset = useCallback(
    async (id: number) => {
      try {
        await DatasetAPI.deleteById(id);
        Taro.showToast({ title: "删除成功", icon: "success" });
        // 刷新列表
        fetchDatasets(1, state.searchKeyword, false);
        // 刷新选项
        fetchDatasetOptions();
        return true;
      } catch (error: any) {
        Taro.showToast({ title: error?.message || "删除失败", icon: "none" });
        return false;
      }
    },
    [fetchDatasets, fetchDatasetOptions, state.searchKeyword]
  );

  // 获取数据集详情
  const fetchDatasetDetail = useCallback(async (datasetId: number) => {
    try {
      dispatch({ type: "SET_DATASET_DETAIL_LOADING", payload: true });
      dispatch({ type: "SET_DATASET_DETAIL_ERROR", payload: null });

      const dataset = await DatasetAPI.getDatasetInfoById(datasetId);
      dispatch({ type: "SET_CURRENT_DATASET", payload: dataset });
    } catch (error: any) {
      dispatch({
        type: "SET_DATASET_DETAIL_ERROR",
        payload: error?.message || "获取数据集详情失败",
      });
    }
  }, []);

  // 获取图片列表（通过数据项接口获取后展平，按标注状态过滤）
  const fetchImages = useCallback(
    async (
      datasetId: number,
      page = 1,
      annotationFilter: AnnotationFilter = "annotated",
      search = "",
      append = false
    ) => {
      try {
        dispatch({ type: "SET_IMAGES_LOADING", payload: true });
        dispatch({ type: "SET_IMAGES_ERROR", payload: null });

        const pageSize = 20;
        const response = await DatasetItemAPI.getList({
          datasetId,
          keyword: search || undefined,
          pageNum: page,
          pageSize,
        });

        const items = (response.list as unknown as DatasetItemVO[]) || [];
        const flattened = flattenItems(items, annotationFilter);
        const total = response.total || 0;
        const hasMore = items.length === pageSize;

        if (append) {
          dispatch({
            type: "APPEND_IMAGES",
            payload: { images: flattened, page, hasMore },
          });
        } else {
          dispatch({
            type: "SET_IMAGES",
            payload: { images: flattened, page, total, hasMore },
          });
        }
      } catch (error: any) {
        dispatch({
          type: "SET_IMAGES_ERROR",
          payload: error?.message || "获取图片列表失败",
        });
      }
    },
    []
  );

  // 其他简化 actions
  const setSearchKeyword = useCallback((keyword: string) => {
    dispatch({ type: "SET_SEARCH_KEYWORD", payload: keyword });
  }, []);

  const setImageSearchKeyword = useCallback((keyword: string) => {
    dispatch({ type: "SET_IMAGE_SEARCH_KEYWORD", payload: keyword });
  }, []);

  const setAnnotationFilter = useCallback((filter: AnnotationFilter) => {
    dispatch({ type: "SET_ANNOTATION_FILTER", payload: filter });
  }, []);

  const setSelectedImage = useCallback((image: ImageUrlVO | null) => {
    dispatch({ type: "SET_SELECTED_IMAGE", payload: image });
  }, []);

  const resetImages = useCallback(() => {
    dispatch({ type: "RESET_IMAGES" });
  }, []);

  const resetState = useCallback(() => {
    dispatch({ type: "RESET_STATE" });
  }, []);

  return {
    state,
    // Actions
    setView,
    setCurrentDatasetId,
    fetchDatasets,
    fetchChildren,
    toggleExpand,
    fetchDatasetOptions,
    createDataset,
    updateDataset,
    deleteDataset,
    fetchDatasetDetail,
    fetchImages,
    setSearchKeyword,
    setImageSearchKeyword,
    setAnnotationFilter,
    setSelectedImage,
    resetImages,
    resetState,
  };
}
