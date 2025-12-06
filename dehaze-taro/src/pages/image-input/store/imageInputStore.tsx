/**
 * 图像输入模块状态管理
 * 使用 Context + Reducer 模式
 */

import { createContext, useContext, useReducer, ReactNode, useCallback } from 'react'
import Taro from '@tarojs/taro'
import {
  InputMethod,
  SampleCategory,
  ImageData,
  SampleImage,
  HistoryRecord,
} from '../services/types'
import { ImageInputService } from '../services/imageInput'
import { HistoryService } from '../services/history'
import { getSampleImagesByCategory } from '../services/sampleData'

// 状态类型定义
interface ImageInputState {
  // 当前输入方式
  activeMethod: InputMethod

  // 当前选中的图片
  currentImage: ImageData | null

  // 上传状态
  uploadProgress: number
  uploadLoading: boolean
  uploadError: string | null

  // 样例图片
  sampleCategory: SampleCategory
  sampleLoading: boolean

  // 历史记录
  historyRecords: HistoryRecord[]
  historyLoading: boolean

  // 预览弹窗
  previewVisible: boolean
}

// 动作类型定义
type ImageInputAction =
  | { type: 'SET_ACTIVE_METHOD'; payload: InputMethod }
  | { type: 'SET_CURRENT_IMAGE'; payload: ImageData | null }
  | { type: 'SET_UPLOAD_PROGRESS'; payload: number }
  | { type: 'SET_UPLOAD_LOADING'; payload: boolean }
  | { type: 'SET_UPLOAD_ERROR'; payload: string | null }
  | { type: 'SET_SAMPLE_CATEGORY'; payload: SampleCategory }
  | { type: 'SET_SAMPLE_LOADING'; payload: boolean }
  | { type: 'SET_HISTORY_RECORDS'; payload: HistoryRecord[] }
  | { type: 'SET_HISTORY_LOADING'; payload: boolean }
  | { type: 'DELETE_HISTORY_RECORD'; payload: number }
  | { type: 'CLEAR_HISTORY' }
  | { type: 'SET_PREVIEW_VISIBLE'; payload: boolean }
  | { type: 'RESET_STATE' }

// 初始状态
const initialState: ImageInputState = {
  activeMethod: 'upload',
  currentImage: null,
  uploadProgress: 0,
  uploadLoading: false,
  uploadError: null,
  sampleCategory: 'all',
  sampleLoading: false,
  historyRecords: [],
  historyLoading: false,
  previewVisible: false,
}

// Reducer
function imageInputReducer(state: ImageInputState, action: ImageInputAction): ImageInputState {
  switch (action.type) {
    case 'SET_ACTIVE_METHOD':
      return { ...state, activeMethod: action.payload }
    case 'SET_CURRENT_IMAGE':
      return { ...state, currentImage: action.payload }
    case 'SET_UPLOAD_PROGRESS':
      return { ...state, uploadProgress: action.payload }
    case 'SET_UPLOAD_LOADING':
      return { ...state, uploadLoading: action.payload }
    case 'SET_UPLOAD_ERROR':
      return { ...state, uploadError: action.payload }
    case 'SET_SAMPLE_CATEGORY':
      return { ...state, sampleCategory: action.payload }
    case 'SET_SAMPLE_LOADING':
      return { ...state, sampleLoading: action.payload }
    case 'SET_HISTORY_RECORDS':
      return { ...state, historyRecords: action.payload, historyLoading: false }
    case 'SET_HISTORY_LOADING':
      return { ...state, historyLoading: action.payload }
    case 'DELETE_HISTORY_RECORD':
      return {
        ...state,
        historyRecords: state.historyRecords.filter(r => r.id !== action.payload),
      }
    case 'CLEAR_HISTORY':
      return { ...state, historyRecords: [] }
    case 'SET_PREVIEW_VISIBLE':
      return { ...state, previewVisible: action.payload }
    case 'RESET_STATE':
      return initialState
    default:
      return state
  }
}

// Context
const ImageInputContext = createContext<{
  state: ImageInputState
  dispatch: React.Dispatch<ImageInputAction>
} | null>(null)

// Provider
export function ImageInputProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(imageInputReducer, initialState)

  return (
    <ImageInputContext.Provider value={{ state, dispatch }}>
      {children}
    </ImageInputContext.Provider>
  )
}

// Hook
export function useImageInput() {
  const context = useContext(ImageInputContext)
  if (!context) {
    throw new Error('useImageInput must be used within an ImageInputProvider')
  }

  const { state, dispatch } = context

  // 切换输入方式
  const setActiveMethod = useCallback((method: InputMethod) => {
    dispatch({ type: 'SET_ACTIVE_METHOD', payload: method })
  }, [])

  // 设置当前图片
  const setCurrentImage = useCallback((image: ImageData | null) => {
    dispatch({ type: 'SET_CURRENT_IMAGE', payload: image })
    if (image) {
      dispatch({ type: 'SET_PREVIEW_VISIBLE', payload: true })
    }
  }, [])

  // 从相册选择图片
  const chooseImageFromAlbum = useCallback(async () => {
    try {
      dispatch({ type: 'SET_UPLOAD_LOADING', payload: true })
      dispatch({ type: 'SET_UPLOAD_ERROR', payload: null })

      const tempFiles = await ImageInputService.chooseImage(1)
      if (tempFiles.length > 0) {
        const imageData = await ImageInputService.processImageFile(tempFiles[0])
        setCurrentImage(imageData)
      }
    } catch (error: any) {
      if (error.code !== 'USER_CANCEL') {
        dispatch({ type: 'SET_UPLOAD_ERROR', payload: error.message })
        Taro.showToast({ title: error.message, icon: 'none' })
      }
    } finally {
      dispatch({ type: 'SET_UPLOAD_LOADING', payload: false })
    }
  }, [setCurrentImage])

  // 拍照
  const takePhoto = useCallback(async () => {
    try {
      dispatch({ type: 'SET_UPLOAD_LOADING', payload: true })
      dispatch({ type: 'SET_UPLOAD_ERROR', payload: null })

      // 检查相机权限
      const hasPermission = await ImageInputService.checkCameraPermission()
      if (!hasPermission) {
        const granted = await ImageInputService.requestCameraPermission()
        if (!granted) {
          throw { message: '相机权限被拒绝' }
        }
      }

      const tempFile = await ImageInputService.takePhoto()
      const imageData = await ImageInputService.processImageFile(tempFile)
      setCurrentImage(imageData)
    } catch (error: any) {
      if (error.code !== 'USER_CANCEL') {
        dispatch({ type: 'SET_UPLOAD_ERROR', payload: error.message })
        Taro.showToast({ title: error.message, icon: 'none' })
      }
    } finally {
      dispatch({ type: 'SET_UPLOAD_LOADING', payload: false })
    }
  }, [setCurrentImage])

  // 选择样例图片
  const selectSampleImage = useCallback(async (sample: SampleImage) => {
    try {
      dispatch({ type: 'SET_SAMPLE_LOADING', payload: true })

      const imageData = await ImageInputService.loadImageFromUrl(sample.url, sample.name)
      setCurrentImage({
        ...imageData,
        sampleInfo: sample,
      })

      Taro.showToast({ title: '样例图片加载成功', icon: 'success' })
    } catch (error: any) {
      Taro.showToast({ title: error.message || '加载失败', icon: 'none' })
    } finally {
      dispatch({ type: 'SET_SAMPLE_LOADING', payload: false })
    }
  }, [setCurrentImage])

  // 切换样例分类
  const setSampleCategory = useCallback((category: SampleCategory) => {
    dispatch({ type: 'SET_SAMPLE_CATEGORY', payload: category })
  }, [])

  // 获取样例图片列表
  const getSampleImages = useCallback((category: SampleCategory) => {
    return getSampleImagesByCategory(category)
  }, [])

  // 加载历史记录
  const loadHistory = useCallback(async () => {
    try {
      dispatch({ type: 'SET_HISTORY_LOADING', payload: true })
      const records = await HistoryService.getHistory()
      dispatch({ type: 'SET_HISTORY_RECORDS', payload: records })
    } catch (error) {
      console.error('加载历史记录失败:', error)
      dispatch({ type: 'SET_HISTORY_RECORDS', payload: [] })
    }
  }, [])

  // 删除历史记录
  const deleteHistoryRecord = useCallback(async (id: number) => {
    try {
      await HistoryService.deleteRecord(id)
      dispatch({ type: 'DELETE_HISTORY_RECORD', payload: id })
      Taro.showToast({ title: '已删除', icon: 'success' })
    } catch (error) {
      Taro.showToast({ title: '删除失败', icon: 'none' })
    }
  }, [])

  // 清空历史记录
  const clearHistory = useCallback(async () => {
    try {
      const res = await Taro.showModal({
        title: '确认清空',
        content: '确定要清空所有历史记录吗？',
        confirmColor: '#ef4444',
      })
      if (res.confirm) {
        await HistoryService.clearHistory()
        dispatch({ type: 'CLEAR_HISTORY' })
        Taro.showToast({ title: '已清空', icon: 'success' })
      }
    } catch (error) {
      Taro.showToast({ title: '清空失败', icon: 'none' })
    }
  }, [])

  // 选择历史记录
  const selectHistoryRecord = useCallback((record: HistoryRecord) => {
    // 从历史记录加载图片
    setCurrentImage({
      url: record.originalImage,
      path: record.originalImage,
      width: 0,
      height: 0,
      size: 0,
      name: record.fileName || '历史图片',
    })
  }, [setCurrentImage])

  // 显示/隐藏预览
  const setPreviewVisible = useCallback((visible: boolean) => {
    dispatch({ type: 'SET_PREVIEW_VISIBLE', payload: visible })
  }, [])

  // 取消选择
  const cancelSelection = useCallback(() => {
    dispatch({ type: 'SET_CURRENT_IMAGE', payload: null })
    dispatch({ type: 'SET_PREVIEW_VISIBLE', payload: false })
  }, [])

  // 确认选择，跳转算法选择页
  const confirmAndNavigate = useCallback(() => {
    if (!state.currentImage) {
      Taro.showToast({ title: '请先选择图片', icon: 'none' })
      return
    }

    // 保存当前图片到全局状态（使用 Taro 的全局变量）
    Taro.setStorageSync('current_image', JSON.stringify(state.currentImage))

    // 跳转到算法选择页面
    Taro.navigateTo({
      url: '/pages/algorithm-select/index',
    })
  }, [state.currentImage])

  // 重置状态
  const resetState = useCallback(() => {
    dispatch({ type: 'RESET_STATE' })
  }, [])

  return {
    state,
    // Actions
    setActiveMethod,
    setCurrentImage,
    chooseImageFromAlbum,
    takePhoto,
    selectSampleImage,
    setSampleCategory,
    getSampleImages,
    loadHistory,
    deleteHistoryRecord,
    clearHistory,
    selectHistoryRecord,
    setPreviewVisible,
    cancelSelection,
    confirmAndNavigate,
    resetState,
  }
}
