/**
 * 图片处理服务
 */

import Taro from '@tarojs/taro'
import {
  ImageData,
  ImageInfo,
  TempFile,
  ImageInputError,
  ErrorCodes,
  ErrorMessages,
  FileSizeLimit,
  SupportedFormats,
} from './types'

// 检查文件格式是否支持
const isSupportedFormat = (path: string): boolean => {
  const ext = path.split('.').pop()?.toLowerCase() || ''
  return SupportedFormats.includes(ext as any)
}

// 获取文件扩展名
const getFileExtension = (path: string): string => {
  return path.split('.').pop()?.toLowerCase() || 'jpg'
}

// 格式化文件大小
export const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) {
    return bytes + ' B'
  } else if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(2) + ' KB'
  } else {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }
}

export const ImageInputService = {
  /**
   * 从相册选择图片
   */
  chooseImage: async (count: number = 1): Promise<TempFile[]> => {
    try {
      const res = await Taro.chooseImage({
        count,
        sizeType: ['original', 'compressed'],
        sourceType: ['album'],
      })

      return res.tempFiles.map(file => ({
        path: file.path,
        size: file.size,
        type: file.type,
      }))
    } catch (error: any) {
      if (error.errMsg?.includes('cancel')) {
        throw { code: 'USER_CANCEL', message: '用户取消选择' } as ImageInputError
      }
      throw {
        code: ErrorCodes.PERMISSION_DENIED,
        message: ErrorMessages[ErrorCodes.PERMISSION_DENIED],
      } as ImageInputError
    }
  },

  /**
   * 拍照
   */
  takePhoto: async (): Promise<TempFile> => {
    try {
      const res = await Taro.chooseImage({
        count: 1,
        sizeType: ['original'],
        sourceType: ['camera'],
      })

      const file = res.tempFiles[0]
      return {
        path: file.path,
        size: file.size,
        type: file.type,
      }
    } catch (error: any) {
      if (error.errMsg?.includes('cancel')) {
        throw { code: 'USER_CANCEL', message: '用户取消拍照' } as ImageInputError
      }
      if (error.errMsg?.includes('auth')) {
        throw {
          code: ErrorCodes.PERMISSION_DENIED,
          message: ErrorMessages[ErrorCodes.PERMISSION_DENIED],
        } as ImageInputError
      }
      throw {
        code: ErrorCodes.CAMERA_NOT_AVAILABLE,
        message: ErrorMessages[ErrorCodes.CAMERA_NOT_AVAILABLE],
      } as ImageInputError
    }
  },

  /**
   * 获取图片信息
   */
  getImageInfo: async (path: string): Promise<ImageInfo> => {
    try {
      const res = await Taro.getImageInfo({ src: path })
      return {
        width: res.width,
        height: res.height,
        path: res.path,
        orientation: res.orientation,
        type: res.type,
      }
    } catch (error) {
      throw {
        code: 'GET_INFO_FAILED',
        message: '获取图片信息失败',
      } as ImageInputError
    }
  },

  /**
   * 压缩图片
   */
  compressImage: async (path: string, quality: number = FileSizeLimit.COMPRESS_QUALITY): Promise<string> => {
    try {
      const res = await Taro.compressImage({
        src: path,
        quality,
      })
      return res.tempFilePath
    } catch (error) {
      console.warn('图片压缩失败，使用原图:', error)
      // 压缩失败时返回原图路径
      return path
    }
  },

  /**
   * 处理选择的图片文件
   * 包含格式检查、大小检查、自动压缩等
   */
  processImageFile: async (tempFile: TempFile): Promise<ImageData> => {
    // 检查文件大小
    if (tempFile.size > FileSizeLimit.MAX_SIZE) {
      throw {
        code: ErrorCodes.FILE_TOO_LARGE,
        message: ErrorMessages[ErrorCodes.FILE_TOO_LARGE],
      } as ImageInputError
    }

    // 获取图片信息
    const imageInfo = await ImageInputService.getImageInfo(tempFile.path)

    let finalPath = tempFile.path
    let compressed = false
    const originalSize = tempFile.size

    // 超过阈值自动压缩
    if (tempFile.size > FileSizeLimit.COMPRESS_THRESHOLD) {
      finalPath = await ImageInputService.compressImage(tempFile.path)
      compressed = finalPath !== tempFile.path
    }

    // 获取压缩后的文件信息（如果压缩了）
    let finalSize = tempFile.size
    if (compressed) {
      try {
        const fileInfo = await Taro.getFileInfo({ filePath: finalPath })
        finalSize = fileInfo.size
      } catch {
        finalSize = originalSize
      }
    }

    // 生成文件名
    const fileName = `image_${Date.now()}.${getFileExtension(tempFile.path)}`

    return {
      url: finalPath,
      path: finalPath,
      width: imageInfo.width,
      height: imageInfo.height,
      size: finalSize,
      name: fileName,
      type: tempFile.type || `image/${getFileExtension(tempFile.path)}`,
      compressed,
      originalSize: compressed ? originalSize : undefined,
    }
  },

  /**
   * 从网络 URL 加载图片
   * 用于加载样例图片
   */
  loadImageFromUrl: async (url: string, name: string): Promise<ImageData> => {
    try {
      // 下载图片到临时文件
      const downloadRes = await Taro.downloadFile({ url })

      if (downloadRes.statusCode !== 200) {
        throw new Error('下载失败')
      }

      const tempPath = downloadRes.tempFilePath

      // 获取图片信息
      const imageInfo = await ImageInputService.getImageInfo(tempPath)

      // 获取文件大小
      let fileSize = 0
      try {
        const fileInfo = await Taro.getFileInfo({ filePath: tempPath })
        fileSize = fileInfo.size
      } catch {
        fileSize = 0
      }

      return {
        url: tempPath,
        path: tempPath,
        width: imageInfo.width,
        height: imageInfo.height,
        size: fileSize,
        name,
        type: 'image/jpeg',
      }
    } catch (error) {
      throw {
        code: ErrorCodes.NETWORK_ERROR,
        message: '样例图片加载失败，请检查网络后重试',
      } as ImageInputError
    }
  },

  /**
   * 检查相机权限
   */
  checkCameraPermission: async (): Promise<boolean> => {
    try {
      const setting = await Taro.getSetting()
      return setting.authSetting['scope.camera'] !== false
    } catch {
      return true
    }
  },

  /**
   * 请求相机权限
   */
  requestCameraPermission: async (): Promise<boolean> => {
    try {
      const res = await Taro.authorize({ scope: 'scope.camera' })
      return true
    } catch {
      // 用户拒绝权限，引导打开设置
      const modalRes = await Taro.showModal({
        title: '需要相机权限',
        content: '请在设置中开启相机权限，用于拍摄需要去雾的图片',
        confirmText: '去设置',
        cancelText: '取消',
      })

      if (modalRes.confirm) {
        Taro.openSetting()
      }
      return false
    }
  },
}
