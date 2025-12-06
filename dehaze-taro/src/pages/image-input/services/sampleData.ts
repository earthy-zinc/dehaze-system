/**
 * 静态样例图片数据
 * 参考 demo/modules/imageInput.js 中的 sampleImages 数据结构
 */

import { SampleImage, SampleCategory } from './types'

// 样例图片数据
export const sampleImages: Record<Exclude<SampleCategory, 'all'>, SampleImage[]> = {
  light: [
    {
      id: 1,
      name: '轻度雾霾-城市街道',
      url: 'https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800',
      category: 'light',
      difficulty: '简单',
      sceneType: '城市',
      recommendAlgorithm: 'DCP'
    },
    {
      id: 2,
      name: '轻度雾霾-公园景观',
      url: 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800',
      category: 'light',
      difficulty: '简单',
      sceneType: '风景',
      recommendAlgorithm: 'AOD-Net'
    },
    {
      id: 3,
      name: '轻度雾霾-建筑物',
      url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800',
      category: 'light',
      difficulty: '简单',
      sceneType: '建筑',
      recommendAlgorithm: 'DCP'
    },
    {
      id: 4,
      name: '轻度雾霾-山景',
      url: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800',
      category: 'light',
      difficulty: '简单',
      sceneType: '风景',
      recommendAlgorithm: 'FFA-Net'
    },
    {
      id: 5,
      name: '轻度雾霾-湖泊',
      url: 'https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=800',
      category: 'light',
      difficulty: '简单',
      sceneType: '风景',
      recommendAlgorithm: 'AOD-Net'
    },
  ],
  medium: [
    {
      id: 6,
      name: '中度雾霾-城市天际线',
      url: 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800',
      category: 'medium',
      difficulty: '中等',
      sceneType: '城市',
      recommendAlgorithm: 'FFA-Net'
    },
    {
      id: 7,
      name: '中度雾霾-道路',
      url: 'https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800',
      category: 'medium',
      difficulty: '中等',
      sceneType: '道路',
      recommendAlgorithm: 'RIDCP'
    },
    {
      id: 8,
      name: '中度雾霾-森林',
      url: 'https://images.unsplash.com/photo-1448375240586-882707db888b?w=800',
      category: 'medium',
      difficulty: '中等',
      sceneType: '风景',
      recommendAlgorithm: 'Dehamer'
    },
    {
      id: 9,
      name: '中度雾霾-海岸',
      url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800',
      category: 'medium',
      difficulty: '中等',
      sceneType: '风景',
      recommendAlgorithm: 'FFA-Net'
    },
    {
      id: 10,
      name: '中度雾霾-乡村',
      url: 'https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800',
      category: 'medium',
      difficulty: '中等',
      sceneType: '乡村',
      recommendAlgorithm: 'AOD-Net'
    },
  ],
  heavy: [
    {
      id: 11,
      name: '重度雾霾-城市中心',
      url: 'https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800',
      category: 'heavy',
      difficulty: '困难',
      sceneType: '城市',
      recommendAlgorithm: 'RIDCP'
    },
    {
      id: 12,
      name: '重度雾霾-高速公路',
      url: 'https://images.unsplash.com/photo-1465447142348-e9952c393450?w=800',
      category: 'heavy',
      difficulty: '困难',
      sceneType: '道路',
      recommendAlgorithm: 'RIDCP'
    },
    {
      id: 13,
      name: '重度雾霾-山区',
      url: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800',
      category: 'heavy',
      difficulty: '困难',
      sceneType: '风景',
      recommendAlgorithm: 'Dehamer'
    },
    {
      id: 14,
      name: '重度雾霾-港口',
      url: 'https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800',
      category: 'heavy',
      difficulty: '困难',
      sceneType: '港口',
      recommendAlgorithm: 'WPXNet'
    },
    {
      id: 15,
      name: '重度雾霾-工业区',
      url: 'https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=800',
      category: 'heavy',
      difficulty: '困难',
      sceneType: '工业',
      recommendAlgorithm: 'RIDCP'
    },
  ],
  special: [
    {
      id: 16,
      name: '特殊场景-夜景雾霾',
      url: 'https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800',
      category: 'special',
      difficulty: '困难',
      sceneType: '夜景',
      recommendAlgorithm: 'Dehamer'
    },
    {
      id: 17,
      name: '特殊场景-逆光雾霾',
      url: 'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800',
      category: 'special',
      difficulty: '困难',
      sceneType: '逆光',
      recommendAlgorithm: 'FFA-Net'
    },
    {
      id: 18,
      name: '特殊场景-雨雾',
      url: 'https://images.unsplash.com/photo-1428908728789-d2de25dbd4e2?w=800',
      category: 'special',
      difficulty: '中等',
      sceneType: '雨雾',
      recommendAlgorithm: 'RIDCP'
    },
    {
      id: 19,
      name: '特殊场景-晨雾',
      url: 'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800',
      category: 'special',
      difficulty: '简单',
      sceneType: '晨雾',
      recommendAlgorithm: 'DCP'
    },
    {
      id: 20,
      name: '特殊场景-雪雾',
      url: 'https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=800',
      category: 'special',
      difficulty: '中等',
      sceneType: '雪景',
      recommendAlgorithm: 'WPXNet'
    },
  ],
}

// 获取所有样例图片
export const getAllSampleImages = (): SampleImage[] => {
  return [
    ...sampleImages.light,
    ...sampleImages.medium,
    ...sampleImages.heavy,
    ...sampleImages.special,
  ]
}

// 按分类获取样例图片
export const getSampleImagesByCategory = (category: SampleCategory): SampleImage[] => {
  if (category === 'all') {
    return getAllSampleImages()
  }
  return sampleImages[category] || []
}

// 分类标签配置
export const categoryTabs = [
  { key: 'all', label: '全部' },
  { key: 'light', label: '轻度雾霾' },
  { key: 'medium', label: '中度雾霾' },
  { key: 'heavy', label: '重度雾霾' },
  { key: 'special', label: '特殊场景' },
]

// 难度对应的样式类名
export const difficultyColorMap: Record<string, string> = {
  '简单': 'success',
  '中等': 'warning',
  '困难': 'danger',
}
