import Taro from '@tarojs/taro'
import {
  Dataset,
  DatasetImage,
  ApiResponse,
  PaginatedResponse,
  DatasetListParams,
  DatasetImageListParams,
  DatasetDetailParams,
} from './types'

// Mock 数据 - 在实际项目中应该替换为真实的 API 调用
const MOCK_DATASETS: Dataset[] = [
  {
    id: 1,
    name: 'RESIDE数据集',
    description: '大规模真实场景图像去雾数据集，包含室内外多种场景',
    creator: 'Li Boyi',
    thumbnail: 'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=400&h=400&fit=crop',
    total_images: 13990,
    foggy_count: 6995,
    clear_count: 6995,
    annotated_count: 0,
    created_at: '2024-01-15T10:30:00Z',
    updated_at: '2024-01-15T10:30:00Z',
  },
  {
    id: 2,
    name: 'O-HAZE数据集',
    description: '户外真实雾霾图像数据集，包含45对有雾/无雾图像',
    creator: 'Ancuti Codruta',
    thumbnail: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop',
    total_images: 90,
    foggy_count: 45,
    clear_count: 45,
    annotated_count: 0,
    created_at: '2024-01-10T14:20:00Z',
    updated_at: '2024-01-10T14:20:00Z',
  },
  {
    id: 3,
    name: 'I-HAZE数据集',
    description: '室内真实雾霾图像数据集，包含35对有雾/无雾图像',
    creator: 'Ancuti Codruta',
    thumbnail: 'https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&h=400&fit=crop',
    total_images: 70,
    foggy_count: 35,
    clear_count: 35,
    annotated_count: 0,
    created_at: '2024-01-08T09:15:00Z',
    updated_at: '2024-01-08T09:15:00Z',
  },
  {
    id: 4,
    name: 'Dense-Haze数据集',
    description: '密集雾霾场景数据集，专注于极端雾霾条件',
    creator: 'Ancuti Codruta',
    thumbnail: 'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400&h=400&fit=crop',
    total_images: 110,
    foggy_count: 55,
    clear_count: 55,
    annotated_count: 0,
    created_at: '2024-01-05T16:45:00Z',
    updated_at: '2024-01-05T16:45:00Z',
  },
  {
    id: 5,
    name: 'NH-HAZE数据集',
    description: '非均匀雾霾数据集，模拟真实世界的复杂雾霾分布',
    creator: 'Ancuti Codruta',
    thumbnail: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop',
    total_images: 110,
    foggy_count: 55,
    clear_count: 55,
    annotated_count: 0,
    created_at: '2024-01-03T11:30:00Z',
    updated_at: '2024-01-03T11:30:00Z',
  },
  {
    id: 6,
    name: 'SOTS数据集',
    description: '合成雾霾数据集，包含室内外场景',
    creator: 'Li Boyi',
    thumbnail: 'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400&h=400&fit=crop',
    total_images: 1000,
    foggy_count: 500,
    clear_count: 500,
    annotated_count: 0,
    created_at: '2024-01-01T08:00:00Z',
    updated_at: '2024-01-01T08:00:00Z',
  },
]

// 生成Mock图片数据
function generateMockImages(datasetId: number, count: number): DatasetImage[] {
  const images: DatasetImage[] = []
  const dataset = MOCK_DATASETS.find(d => d.id === datasetId)
  if (!dataset) return images

  const imageTypes: Array<'foggy' | 'clear' | 'annotated'> = ['foggy', 'clear', 'annotated']
  const sampleImages = [
    'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&h=600&fit=crop',
  ]

  for (let i = 0; i < count; i++) {
    const type = imageTypes[i % 3]
    const typeCount = type === 'foggy'
      ? dataset.foggy_count
      : type === 'clear'
        ? dataset.clear_count
        : dataset.annotated_count

    if (typeCount === 0) continue

    images.push({
      id: datasetId * 1000 + i,
      dataset_id: datasetId,
      filename: `${dataset.name.replace(/\s+/g, '_')}_${type}_${String(i + 1).padStart(4, '0')}.jpg`,
      image_url: sampleImages[i % sampleImages.length],
      image_type: type,
      width: 1920,
      height: 1080,
      file_size: Math.floor(Math.random() * 2000000) + 500000,
      tags: `${type},${dataset.name}`,
      description: `${dataset.name}中的${type === 'foggy' ? '有雾' : type === 'clear' ? '无雾' : '标注'}图像`,
      created_at: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
    })
  }

  return images
}

// 模拟网络延迟
const delay = (ms: number = 300) => new Promise(resolve => setTimeout(resolve, ms))

// API 服务类
export class DatasetService {
  // 获取数据集列表
  static async getDatasetList(params: DatasetListParams = {}): Promise<ApiResponse<PaginatedResponse<Dataset>>> {
    await delay(300)

    let filteredDatasets = [...MOCK_DATASETS]

    // 搜索过滤
    if (params.search) {
      const keyword = params.search.toLowerCase()
      filteredDatasets = filteredDatasets.filter(
        d => d.name.toLowerCase().includes(keyword) ||
             (d.description && d.description.toLowerCase().includes(keyword))
      )
    }

    const pageSize = params.page_size || 10
    const page = params.page || 1
    const start = (page - 1) * pageSize
    const end = start + pageSize
    const list = filteredDatasets.slice(start, end)

    return {
      code: 0,
      data: {
        list,
        total: filteredDatasets.length,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(filteredDatasets.length / pageSize),
      },
    }
  }

  // 获取数据集详情
  static async getDatasetDetail(params: DatasetDetailParams): Promise<ApiResponse<Dataset>> {
    await delay(200)

    const dataset = MOCK_DATASETS.find(d => d.id === params.id)

    if (!dataset) {
      return {
        code: 404,
        message: '数据集不存在',
        data: null as any,
      }
    }

    return {
      code: 0,
      data: dataset,
    }
  }

  // 获取数据集图片列表
  static async getDatasetImages(params: DatasetImageListParams): Promise<ApiResponse<PaginatedResponse<DatasetImage>>> {
    await delay(400)

    // 生成该数据集的所有图片
    let allImages = generateMockImages(params.dataset_id, 60)

    // 类型过滤
    if (params.image_type && params.image_type !== 'all') {
      allImages = allImages.filter(img => img.image_type === params.image_type)
    }

    // 搜索过滤
    if (params.search) {
      const keyword = params.search.toLowerCase()
      allImages = allImages.filter(
        img => img.filename.toLowerCase().includes(keyword) ||
               (img.tags && img.tags.toLowerCase().includes(keyword)) ||
               (img.description && img.description.toLowerCase().includes(keyword))
      )
    }

    const pageSize = params.page_size || 20
    const page = params.page || 1
    const start = (page - 1) * pageSize
    const end = start + pageSize
    const list = allImages.slice(start, end)

    return {
      code: 0,
      data: {
        list,
        total: allImages.length,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(allImages.length / pageSize),
      },
    }
  }
}