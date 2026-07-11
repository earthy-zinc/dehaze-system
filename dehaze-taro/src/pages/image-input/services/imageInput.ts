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
  MinResolution,
} from './types'

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

// 判断错误是否为 ImageInputError
const isImageInputError = (err: any): err is ImageInputError => {
  return err instanceof ImageInputError || (err && typeof err.code === 'string' && err.code in ErrorCodes)
}

export const ImageInputService = {
  /**
   * 从相册选择图片
   */
  chooseImage: async (count: number = 1): Promise<TempFile[]> => {
    try {
      const res = await Taro.chooseMedia({
        count,
        mediaType: ['image'],
        sourceType: ['album'],
        sizeType: ['original', 'compressed'],
      })

      return res.tempFiles.map(file => ({
        path: file.tempFilePath,
        size: file.size,
        type: file.fileType,
      }))
    } catch (error: any) {
      if (error.errMsg?.includes('cancel')) {
        throw new ImageInputError(ErrorCodes.USER_CANCEL, ErrorMessages[ErrorCodes.USER_CANCEL])
      }
      throw new ImageInputError(
        ErrorCodes.PERMISSION_DENIED,
        ErrorMessages[ErrorCodes.PERMISSION_DENIED]
      )
    }
  },

  /**
   * 拍照
   */
  takePhoto: async (): Promise<TempFile> => {
    try {
      const res = await Taro.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sourceType: ['camera'],
        sizeType: ['original'],
      })

      const file = res.tempFiles[0]
      return {
        path: file.tempFilePath,
        size: file.size,
        type: file.fileType,
      }
    } catch (error: any) {
      if (error.errMsg?.includes('cancel')) {
        throw new ImageInputError(ErrorCodes.USER_CANCEL, '用户取消拍照')
      }
      if (error.errMsg?.includes('auth')) {
        throw new ImageInputError(
          ErrorCodes.PERMISSION_DENIED,
          ErrorMessages[ErrorCodes.PERMISSION_DENIED]
        )
      }
      throw new ImageInputError(
        ErrorCodes.CAMERA_NOT_AVAILABLE,
        ErrorMessages[ErrorCodes.CAMERA_NOT_AVAILABLE]
      )
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
      throw new ImageInputError('GET_INFO_FAILED', '获取图片信息失败')
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
   * 包含格式校验、大小检查、分辨率校验、自动压缩
   */
  processImageFile: async (tempFile: TempFile): Promise<ImageData> => {
    // 格式校验
    const ext = getFileExtension(tempFile.path)
    if (!SupportedFormats.includes(ext as any)) {
      throw new ImageInputError(
        ErrorCodes.UNSUPPORTED_FORMAT,
        ErrorMessages[ErrorCodes.UNSUPPORTED_FORMAT]
      )
    }

    // 检查文件大小
    if (tempFile.size > FileSizeLimit.MAX_SIZE) {
      throw new ImageInputError(
        ErrorCodes.FILE_TOO_LARGE,
        ErrorMessages[ErrorCodes.FILE_TOO_LARGE]
      )
    }

    // 获取图片信息
    const imageInfo = await ImageInputService.getImageInfo(tempFile.path)

    // 分辨率校验（低于最低要求时警告，但不阻止）
    if (imageInfo.width < MinResolution.WIDTH || imageInfo.height < MinResolution.HEIGHT) {
      Taro.showToast({ title: ErrorMessages[ErrorCodes.RESOLUTION_LOW], icon: 'none', duration: 2000 })
    }

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
        if ('size' in fileInfo) {
          finalSize = fileInfo.size
        } else {
          finalSize = originalSize
        }
      } catch {
        finalSize = originalSize
      }
    }

    // 生成文件名
    const fileName = `image_${Date.now()}.${ext}`

    return {
      url: finalPath,
      path: finalPath,
      width: imageInfo.width,
      height: imageInfo.height,
      size: finalSize,
      name: fileName,
      type: tempFile.type || `image/${ext}`,
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
        if ('size' in fileInfo) {
          fileSize = fileInfo.size
        }
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
      throw new ImageInputError(
        ErrorCodes.NETWORK_ERROR,
        '样例图片加载失败，请检查网络后重试'
      )
    }
  },

  /**
   * 检查相机权限
   */
  checkCameraPermission: async (): Promise<boolean> => {
    try {
      const setting = await Taro.getSetting()
      return setting.authSetting['scope.camera'] === true
    } catch {
      return false
    }
  },

  /**
   * 请求相机权限
   */
  requestCameraPermission: async (): Promise<boolean> => {
    try {
      await Taro.authorize({ scope: 'scope.camera' })
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

// 导出错误判断工具供 store 使用
export { isImageInputError }
