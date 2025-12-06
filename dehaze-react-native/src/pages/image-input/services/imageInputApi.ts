/**
 * 图像输入模块 API 服务
 */

import { Image } from 'react-native';
import {
  SampleImage,
  SampleCategory,
  ValidationResult,
  ImageInfo,
  SampleListResponse,
} from '../types/imageInput';

// 支持的图片格式
const SUPPORTED_FORMATS = ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'];

// 文件大小限制
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const COMPRESSION_THRESHOLD = 5 * 1024 * 1024; // 5MB

// Mock 样例图片数据
const MOCK_SAMPLES: SampleImage[] = [
  // 轻度雾霾
  {
    id: 1,
    name: '轻度雾霾-城市街道',
    url: 'https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800',
    category: 'light',
    difficulty: 'easy',
    sceneType: '城市',
    recommendedAlgorithm: 'DCP',
  },
  {
    id: 2,
    name: '轻度雾霾-公园景观',
    url: 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800',
    category: 'light',
    difficulty: 'easy',
    sceneType: '自然',
    recommendedAlgorithm: 'DCP',
  },
  {
    id: 3,
    name: '轻度雾霾-建筑物',
    url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800',
    category: 'light',
    difficulty: 'easy',
    sceneType: '建筑',
    recommendedAlgorithm: 'CAP',
  },
  {
    id: 4,
    name: '轻度雾霾-山景',
    url: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800',
    category: 'light',
    difficulty: 'easy',
    sceneType: '山景',
    recommendedAlgorithm: 'DCP',
  },
  {
    id: 5,
    name: '轻度雾霾-湖泊',
    url: 'https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=800',
    category: 'light',
    difficulty: 'easy',
    sceneType: '水景',
    recommendedAlgorithm: 'CAP',
  },
  // 中度雾霾
  {
    id: 6,
    name: '中度雾霾-城市天际线',
    url: 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800',
    category: 'medium',
    difficulty: 'medium',
    sceneType: '城市',
    recommendedAlgorithm: 'AOD-Net',
  },
  {
    id: 7,
    name: '中度雾霾-道路',
    url: 'https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800',
    category: 'medium',
    difficulty: 'medium',
    sceneType: '道路',
    recommendedAlgorithm: 'DehazeNet',
  },
  {
    id: 8,
    name: '中度雾霾-森林',
    url: 'https://images.unsplash.com/photo-1448375240586-882707db888b?w=800',
    category: 'medium',
    difficulty: 'medium',
    sceneType: '森林',
    recommendedAlgorithm: 'AOD-Net',
  },
  {
    id: 9,
    name: '中度雾霾-海岸',
    url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800',
    category: 'medium',
    difficulty: 'medium',
    sceneType: '海岸',
    recommendedAlgorithm: 'FFA-Net',
  },
  {
    id: 10,
    name: '中度雾霾-乡村',
    url: 'https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800',
    category: 'medium',
    difficulty: 'medium',
    sceneType: '乡村',
    recommendedAlgorithm: 'DehazeNet',
  },
  // 重度雾霾
  {
    id: 11,
    name: '重度雾霾-城市中心',
    url: 'https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800',
    category: 'heavy',
    difficulty: 'hard',
    sceneType: '城市',
    recommendedAlgorithm: 'MSBDN',
  },
  {
    id: 12,
    name: '重度雾霾-高速公路',
    url: 'https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=800',
    category: 'heavy',
    difficulty: 'hard',
    sceneType: '道路',
    recommendedAlgorithm: 'GridDehazeNet',
  },
  {
    id: 13,
    name: '重度雾霾-山区',
    url: 'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800',
    category: 'heavy',
    difficulty: 'hard',
    sceneType: '山区',
    recommendedAlgorithm: 'MSBDN',
  },
  {
    id: 14,
    name: '重度雾霾-港口',
    url: 'https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800',
    category: 'heavy',
    difficulty: 'hard',
    sceneType: '港口',
    recommendedAlgorithm: 'FFA-Net',
  },
  {
    id: 15,
    name: '重度雾霾-工业区',
    url: 'https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=800',
    category: 'heavy',
    difficulty: 'hard',
    sceneType: '工业',
    recommendedAlgorithm: 'GridDehazeNet',
  },
  // 特殊场景
  {
    id: 16,
    name: '特殊场景-夜景雾霾',
    url: 'https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800',
    category: 'special',
    difficulty: 'hard',
    sceneType: '夜景',
    recommendedAlgorithm: 'MSBDN',
  },
  {
    id: 17,
    name: '特殊场景-逆光雾霾',
    url: 'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800',
    category: 'special',
    difficulty: 'hard',
    sceneType: '逆光',
    recommendedAlgorithm: 'FFA-Net',
  },
  {
    id: 18,
    name: '特殊场景-雨雾',
    url: 'https://images.unsplash.com/photo-1428908728789-d2de25dbd4e2?w=800',
    category: 'special',
    difficulty: 'medium',
    sceneType: '雨天',
    recommendedAlgorithm: 'AOD-Net',
  },
  {
    id: 19,
    name: '特殊场景-晨雾',
    url: 'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800',
    category: 'special',
    difficulty: 'easy',
    sceneType: '晨雾',
    recommendedAlgorithm: 'DCP',
  },
  {
    id: 20,
    name: '特殊场景-雪雾',
    url: 'https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=800',
    category: 'special',
    difficulty: 'medium',
    sceneType: '雪景',
    recommendedAlgorithm: 'DehazeNet',
  },
];

// 模拟网络延迟
const delay = (ms: number = 300) => new Promise<void>(resolve => setTimeout(resolve, ms));

export const imageInputApi = {
  /**
   * 获取样例图片列表
   */
  fetchSamples: async (category?: SampleCategory): Promise<SampleListResponse> => {
    await delay(300);

    let filteredSamples = [...MOCK_SAMPLES];

    if (category && category !== 'all') {
      filteredSamples = filteredSamples.filter(s => s.category === category);
    }

    return {
      code: 0,
      data: {
        list: filteredSamples,
        total: filteredSamples.length,
      },
    };
  },

  /**
   * 验证图片格式和大小
   */
  validateImage: (fileSize: number, type?: string): ValidationResult => {
    // 检查文件大小
    if (fileSize > MAX_FILE_SIZE) {
      return {
        valid: false,
        error: '图片大小超过20MB，请选择较小的图片',
      };
    }

    // 检查格式（如果提供了类型）
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
   * 获取随机样例图片（用于快速体验）
   */
  getRandomSample: (): SampleImage => {
    const randomIndex = Math.floor(Math.random() * MOCK_SAMPLES.length);
    return MOCK_SAMPLES[randomIndex];
  },

  /**
   * 根据难度获取样例图片
   */
  getSamplesByDifficulty: (difficulty: 'easy' | 'medium' | 'hard'): SampleImage[] => {
    return MOCK_SAMPLES.filter(s => s.difficulty === difficulty);
  },
};

export { MOCK_SAMPLES, MAX_FILE_SIZE, COMPRESSION_THRESHOLD };
