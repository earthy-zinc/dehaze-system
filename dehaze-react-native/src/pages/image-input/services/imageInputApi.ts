/**
 * 图像输入模块 API 服务
 *
 * 样例库通过 SDK 的 DatasetItemAPI 从后端数据集项获取（取 hazyImages）。
 */

import { Image } from 'react-native';
import { DatasetItemAPI, FileAPI } from 'dehaze-sdk-js';
import type { DatasetItemVO, FileInfo } from 'dehaze-sdk-js';
import {
  SampleImage,
  SampleCategory,
  DifficultyLevel,
  ValidationResult,
} from '../types/imageInput';

// 支持的图片格式
const SUPPORTED_FORMATS = ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'];

// 文件大小限制
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const COMPRESSION_THRESHOLD = 5 * 1024 * 1024; // 5MB

/** hazeLevel 到 SampleCategory 的映射 */
function hazeLevelToCategory(hazeLevel?: string): SampleCategory {
  if (hazeLevel === 'light' || hazeLevel === 'medium' || hazeLevel === 'heavy') {
    return hazeLevel;
  }
  return 'medium';
}

/** hazeLevel 到 DifficultyLevel 的映射 */
function hazeLevelToDifficulty(hazeLevel?: string): DifficultyLevel {
  switch (hazeLevel) {
    case 'light':
      return 'easy';
    case 'heavy':
      return 'hard';
    default:
      return 'medium';
  }
}

/** 将后端 DatasetItemVO 列表映射为样例图片列表（展开 hazyImages） */
function mapToSamples(items: DatasetItemVO[]): SampleImage[] {
  const samples: SampleImage[] = [];
  for (const item of items) {
    if (!item.hazyImages || item.hazyImages.length === 0) continue;
    for (const img of item.hazyImages) {
      samples.push({
        id: img.id,
        name: img.fileName || item.name,
        url: img.url,
        thumbUrl: img.thumbnailUrl,
        category: hazeLevelToCategory(img.hazeLevel),
        difficulty: hazeLevelToDifficulty(img.hazeLevel),
        sceneType: img.sceneType || item.sceneType,
        width: img.width,
        height: img.height,
      });
    }
  }
  return samples;
}

export const imageInputApi = {
  /**
   * 获取样例图片列表（从后端数据集项的 hazyImages 获取）
   */
  fetchSamples: async (category?: SampleCategory): Promise<SampleImage[]> => {
    const hazeLevel = category && category !== 'all' ? category : undefined;
    const result = await DatasetItemAPI.getList({
      pageNum: 1,
      pageSize: 100,
      hazeLevel,
    });
    const samples = mapToSamples(result.list);
    // 后端按 item 级过滤，二次按 hazeLevel 精确过滤
    if (hazeLevel) {
      return samples.filter(s => s.category === hazeLevel);
    }
    return samples;
  },

  /**
   * 验证图片格式和大小
   */
  validateImage: (fileSize: number, type?: string): ValidationResult => {
    if (fileSize > MAX_FILE_SIZE) {
      return {
        valid: false,
        error: '图片大小超过20MB，请选择较小的图片',
      };
    }

    if (type && !SUPPORTED_FORMATS.includes(type.toLowerCase())) {
      return {
        valid: false,
        error: '不支持该图片格式，请选择JPG/PNG/WEBP/HEIC格式',
      };
    }

    return { valid: true };
  },

  /**
   * 检查是否需要压缩
   */
  needsCompression: (fileSize: number): boolean => {
    return fileSize > COMPRESSION_THRESHOLD;
  },

  /**
   * 获取图片尺寸信息
   */
  getImageSize: (uri: string): Promise<{ width: number; height: number }> => {
    return new Promise((resolve, reject) => {
      Image.getSize(
        uri,
        (width, height) => resolve({ width, height }),
        (error) => reject(error)
      );
    });
  },

  /**
   * 格式化文件大小
   */
  formatFileSize: (bytes: number): string => {
    if (bytes < 1024) {
      return `${bytes} B`;
    } else if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    } else {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
  },

  /**
   * 上传本地图片到后端文件服务（/api/v1/files）
   *
   * React Native 中通过 FormData 文件描述符 {uri, name, type} 上传，
   * 返回的 FileInfo.url 为后端可访问的远程地址，用于后续预测/评估。
   */
  uploadImage: async (
    uri: string,
    fileName: string,
    fileType?: string,
  ): Promise<FileInfo> => {
    // RN 的 FormData 接受 {uri, name, type} 形式的文件描述符
    const fileDescriptor = {
      uri,
      name: fileName,
      type: fileType || 'image/jpeg',
    } as unknown as File;
    return await FileAPI.upload(fileDescriptor);
  },

  /**
   * 随机获取一张样例图片（用于快速体验）
   */
  getRandomSample: async (): Promise<SampleImage> => {
    const samples = await imageInputApi.fetchSamples();
    if (samples.length === 0) {
      throw new Error('暂无可用样例图片');
    }
    const randomIndex = Math.floor(Math.random() * samples.length);
    return samples[randomIndex];
  },
};

export { MAX_FILE_SIZE, COMPRESSION_THRESHOLD };
