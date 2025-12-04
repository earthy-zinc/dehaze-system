import { createContext, useContext, useReducer, ReactNode, useCallback, useEffect } from 'react'
import { Dataset, DatasetImage } from '../services/types'
import { DatasetService } from '../services/dataset'

// 状态类型定义
interface DatasetState {
  // 视图状态
  currentView: 'list' | 'detail'
  currentDatasetId: number | null

  // 数据集列表
  datasets: Dataset[]
  datasetsLoading: boolean
  datasetsError: string | null
  datasetsPage: number
  datasetsHasMore: boolean
  datasetsTotal: number

  // 当前数据集详情
  currentDataset: Dataset | null
  datasetDetailLoading: boolean
  datasetDetailError: string | null

  // 图片列表
  images: DatasetImage[]
  imagesLoading: boolean
  imagesError: string | null
  imagesPage: number
  imagesHasMore: boolean
  imagesTotal: number
  currentImageType: 'all' | 'foggy' | 'clear' | 'annotated'

  // 搜索状态
  searchKeyword: string
  imageSearchKeyword: string

  // 选中的图片（用于查看器）
  selectedImage: DatasetImage | null
}

// 动作类型定义
type DatasetAction =
  | { type: 'SET_VIEW'; payload: 'list' | 'detail' }
  | { type: 'SET_CURRENT_DATASET_ID'; payload: number | null }
  | { type: 'SET_DATASETS_LOADING'; payload: boolean }
  | { type: 'SET_DATASETS_ERROR'; payload: string | null }
  | { type: 'SET_DATASETS'; payload: { datasets: Dataset[]; page: number; total: number; hasMore: boolean } }
  | { type: 'APPEND_DATASETS'; payload: { datasets: Dataset[]; page: number; hasMore: boolean } }
  | { type: 'SET_DATASET_DETAIL_LOADING'; payload: boolean }
  | { type: 'SET_DATASET_DETAIL_ERROR'; payload: string | null }
  | { type: 'SET_CURRENT_DATASET'; payload: Dataset | null }
  | { type: 'SET_IMAGES_LOADING'; payload: boolean }
  | { type: 'SET_IMAGES_ERROR'; payload: string | null }
  | { type: 'SET_IMAGES'; payload: { images: DatasetImage[]; page: number; total: number; hasMore: boolean } }
  | { type: 'APPEND_IMAGES'; payload: { images: DatasetImage[]; page: number; hasMore: boolean } }
  | { type: 'SET_IMAGE_TYPE'; payload: 'all' | 'foggy' | 'clear' | 'annotated' }
  | { type: 'SET_SEARCH_KEYWORD'; payload: string }
  | { type: 'SET_IMAGE_SEARCH_KEYWORD'; payload: string }
  | { type: 'SET_SELECTED_IMAGE'; payload: DatasetImage | null }
  | { type: 'RESET_IMAGES' }
  | { type: 'RESET_STATE' }

// 初始状态
const initialState: DatasetState = {
  currentView: 'list',
  currentDatasetId: null,
  datasets: [],
  datasetsLoading: false,
  datasetsError: null,
  datasetsPage: 1,
  datasetsHasMore: true,
  datasetsTotal: 0,
  currentDataset: null,
  datasetDetailLoading: false,
  datasetDetailError: null,
  images: [],
  imagesLoading: false,
  imagesError: null,
  imagesPage: 1,
  imagesHasMore: true,
  imagesTotal: 0,
  currentImageType: 'all',
  searchKeyword: '',
  imageSearchKeyword: '',
  selectedImage: null,
}

// Reducer
function datasetReducer(state: DatasetState, action: DatasetAction): DatasetState {
  switch (action.type) {
    case 'SET_VIEW':
      return { ...state, currentView: action.payload }
    case 'SET_CURRENT_DATASET_ID':
      return { ...state, currentDatasetId: action.payload }
    case 'SET_DATASETS_LOADING':
      return { ...state, datasetsLoading: action.payload }
    case 'SET_DATASETS_ERROR':
      return { ...state, datasetsError: action.payload }
    case 'SET_DATASETS':
      return {
        ...state,
        datasets: action.payload.datasets,
        datasetsPage: action.payload.page,
        datasetsTotal: action.payload.total,
        datasetsHasMore: action.payload.hasMore,
        datasetsLoading: false,
        datasetsError: null,
      }
    case 'APPEND_DATASETS':
      return {
        ...state,
        datasets: [...state.datasets, ...action.payload.datasets],
        datasetsPage: action.payload.page,
        datasetsHasMore: action.payload.hasMore,
        datasetsLoading: false,
      }
    case 'SET_DATASET_DETAIL_LOADING':
      return { ...state, datasetDetailLoading: action.payload }
    case 'SET_DATASET_DETAIL_ERROR':
      return { ...state, datasetDetailError: action.payload }
    case 'SET_CURRENT_DATASET':
      return { ...state, currentDataset: action.payload, datasetDetailLoading: false, datasetDetailError: null }
    case 'SET_IMAGES_LOADING':
      return { ...state, imagesLoading: action.payload }
    case 'SET_IMAGES_ERROR':
      return { ...state, imagesError: action.payload }
    case 'SET_IMAGES':
      return {
        ...state,
        images: action.payload.images,
        imagesPage: action.payload.page,
        imagesTotal: action.payload.total,
        imagesHasMore: action.payload.hasMore,
        imagesLoading: false,
        imagesError: null,
      }
    case 'APPEND_IMAGES':
      return {
        ...state,
        images: [...state.images, ...action.payload.images],
        imagesPage: action.payload.page,
        imagesHasMore: action.payload.hasMore,
        imagesLoading: false,
      }
    case 'SET_IMAGE_TYPE':
      return { ...state, currentImageType: action.payload }
    case 'SET_SEARCH_KEYWORD':
      return { ...state, searchKeyword: action.payload }
    case 'SET_IMAGE_SEARCH_KEYWORD':
      return { ...state, imageSearchKeyword: action.payload }
    case 'SET_SELECTED_IMAGE':
      return { ...state, selectedImage: action.payload }
    case 'RESET_IMAGES':
      return {
        ...state,
        images: [],
        imagesPage: 1,
        imagesHasMore: true,
        imagesTotal: 0,
        imageSearchKeyword: '',
        currentImageType: 'all',
      }
    case 'RESET_STATE':
      return initialState
    default:
      return state
  }
}

// Context
const DatasetContext = createContext<{
  state: DatasetState
  dispatch: React.Dispatch<DatasetAction>
} | null>(null)

// Provider
export function DatasetProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(datasetReducer, initialState)

  return (
    <DatasetContext.Provider value={{ state, dispatch }}>
      {children}
    </DatasetContext.Provider>
  )
}

// Hook
export function useDataset() {
  const context = useContext(DatasetContext)
  if (!context) {
    throw new Error('useDataset must be used within a DatasetProvider')
  }

  const { state, dispatch } = context

  // Actions
  const setView = useCallback((view: 'list' | 'detail') => {
    dispatch({ type: 'SET_VIEW', payload: view })
  }, [])

  const setCurrentDatasetId = useCallback((id: number | null) => {
    dispatch({ type: 'SET_CURRENT_DATASET_ID', payload: id })
  }, [])

  // 获取数据集列表
  const fetchDatasets = useCallback(async (page = 1, search = '', append = false) => {
    try {
      dispatch({ type: 'SET_DATASETS_LOADING', payload: true })
      dispatch({ type: 'SET_DATASETS_ERROR', payload: null })

      const response = await DatasetService.getDatasetList({
        page,
        page_size: 10,
        search,
      })

      if (response.code === 0) {
        if (append) {
          dispatch({
            type: 'APPEND_DATASETS',
            payload: {
              datasets: response.data.list,
              page: response.data.page,
              hasMore: response.data.page < response.data.total_pages,
            },
          })
        } else {
          dispatch({
            type: 'SET_DATASETS',
            payload: {
              datasets: response.data.list,
              page: response.data.page,
              total: response.data.total,
              hasMore: response.data.page < response.data.total_pages,
            },
          })
        }
      } else {
        dispatch({ type: 'SET_DATASETS_ERROR', payload: response.message || '获取数据集列表失败' })
      }
    } catch (error) {
      dispatch({ type: 'SET_DATASETS_ERROR', payload: '网络错误，请重试' })
    }
  }, [])

  // 获取数据集详情
  const fetchDatasetDetail = useCallback(async (datasetId: number) => {
    try {
      dispatch({ type: 'SET_DATASET_DETAIL_LOADING', payload: true })
      dispatch({ type: 'SET_DATASET_DETAIL_ERROR', payload: null })

      const response = await DatasetService.getDatasetDetail({ id: datasetId })

      if (response.code === 0) {
        dispatch({ type: 'SET_CURRENT_DATASET', payload: response.data })
      } else {
        dispatch({ type: 'SET_DATASET_DETAIL_ERROR', payload: response.message || '获取数据集详情失败' })
      }
    } catch (error) {
      dispatch({ type: 'SET_DATASET_DETAIL_ERROR', payload: '网络错误，请重试' })
    }
  }, [])

  // 获取图片列表
  const fetchImages = useCallback(async (datasetId: number, page = 1, imageType = 'all' as const, search = '', append = false) => {
    try {
      dispatch({ type: 'SET_IMAGES_LOADING', payload: true })
      dispatch({ type: 'SET_IMAGES_ERROR', payload: null })

      const response = await DatasetService.getDatasetImages({
        dataset_id: datasetId,
        page,
        page_size: 20,
        image_type,
        search,
      })

      if (response.code === 0) {
        if (append) {
          dispatch({
            type: 'APPEND_IMAGES',
            payload: {
              images: response.data.list,
              page: response.data.page,
              hasMore: response.data.page < response.data.total_pages,
            },
          })
        } else {
          dispatch({
            type: 'SET_IMAGES',
            payload: {
              images: response.data.list,
              page: response.data.page,
              total: response.data.total,
              hasMore: response.data.page < response.data.total_pages,
            },
          })
        }
      } else {
        dispatch({ type: 'SET_IMAGES_ERROR', payload: response.message || '获取图片列表失败' })
      }
    } catch (error) {
      dispatch({ type: 'SET_IMAGES_ERROR', payload: '网络错误，请重试' })
    }
  }, [])

  // 其他简化 actions
  const setSearchKeyword = useCallback((keyword: string) => {
    dispatch({ type: 'SET_SEARCH_KEYWORD', payload: keyword })
  }, [])

  const setImageSearchKeyword = useCallback((keyword: string) => {
    dispatch({ type: 'SET_IMAGE_SEARCH_KEYWORD', payload: keyword })
  }, [])

  const setImageType = useCallback((type: 'all' | 'foggy' | 'clear' | 'annotated') => {
    dispatch({ type: 'SET_IMAGE_TYPE', payload: type })
  }, [])

  const setSelectedImage = useCallback((image: DatasetImage | null) => {
    dispatch({ type: 'SET_SELECTED_IMAGE', payload: image })
  }, [])

  const resetImages = useCallback(() => {
    dispatch({ type: 'RESET_IMAGES' })
  }, [])

  const resetState = useCallback(() => {
    dispatch({ type: 'RESET_STATE' })
  }, [])

  return {
    state,
    // Actions
    setView,
    setCurrentDatasetId,
    fetchDatasets,
    fetchDatasetDetail,
    fetchImages,
    setSearchKeyword,
    setImageSearchKeyword,
    setImageType,
    setSelectedImage,
    resetImages,
    resetState,
  }
}