/**
 * 历史记录服务
 * 使用本地存储实现，预留云端同步扩展接口
 */

import Taro from '@tarojs/taro'
import { HistoryRecord, IHistoryStorage, GroupedHistory } from './types'

const STORAGE_KEY = 'dehaze_history'
const MAX_RECORDS = 20

/**
 * 本地存储实现
 */
class LocalHistoryStorage implements IHistoryStorage {
  private storageKey: string
  private maxRecords: number

  constructor(storageKey: string = STORAGE_KEY, maxRecords: number = MAX_RECORDS) {
    this.storageKey = storageKey
    this.maxRecords = maxRecords
  }

  async getHistory(): Promise<HistoryRecord[]> {
    try {
      const data = Taro.getStorageSync(this.storageKey)
      return data ? JSON.parse(data) : []
    } catch (error) {
      console.error('获取历史记录失败:', error)
      return []
    }
  }

  async addRecord(record: Omit<HistoryRecord, 'id'>): Promise<void> {
    try {
      const history = await this.getHistory()
      const newRecord: HistoryRecord = {
        ...record,
        id: Date.now(),
      }
      history.unshift(newRecord)

      // 限制记录数量
      if (history.length > this.maxRecords) {
        history.splice(this.maxRecords)
      }

      Taro.setStorageSync(this.storageKey, JSON.stringify(history))
    } catch (error) {
      console.error('添加历史记录失败:', error)
    }
  }

  async deleteRecord(id: number): Promise<void> {
    try {
      const history = await this.getHistory()
      const filtered = history.filter(record => record.id !== id)
      Taro.setStorageSync(this.storageKey, JSON.stringify(filtered))
    } catch (error) {
      console.error('删除历史记录失败:', error)
    }
  }

  async clearHistory(): Promise<void> {
    try {
      Taro.removeStorageSync(this.storageKey)
    } catch (error) {
      console.error('清空历史记录失败:', error)
    }
  }
}

// 导出单例
export const HistoryService = new LocalHistoryStorage()

/**
 * 将历史记录按时间分组
 */
export const groupHistoryByDate = (records: HistoryRecord[]): GroupedHistory[] => {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000)
  const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)

  const groups: Record<string, HistoryRecord[]> = {
    '今天': [],
    '昨天': [],
    '最近7天': [],
    '更早': [],
  }

  records.forEach(record => {
    const recordDate = new Date(record.timestamp)
    const recordDay = new Date(recordDate.getFullYear(), recordDate.getMonth(), recordDate.getDate())

    if (recordDay.getTime() === today.getTime()) {
      groups['今天'].push(record)
    } else if (recordDay.getTime() === yesterday.getTime()) {
      groups['昨天'].push(record)
    } else if (recordDay.getTime() > lastWeek.getTime()) {
      groups['最近7天'].push(record)
    } else {
      groups['更早'].push(record)
    }
  })

  // 过滤空分组
  return Object.entries(groups)
    .filter(([_, records]) => records.length > 0)
    .map(([title, records]) => ({ title, records }))
}

/**
 * 格式化时间显示
 */
export const formatTimestamp = (timestamp: string): string => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  if (diff < 60 * 1000) {
    return '刚刚'
  } else if (diff < 60 * 60 * 1000) {
    return `${Math.floor(diff / (60 * 1000))}分钟前`
  } else if (diff < 24 * 60 * 60 * 1000) {
    return `${Math.floor(diff / (60 * 60 * 1000))}小时前`
  } else if (diff < 7 * 24 * 60 * 60 * 1000) {
    return `${Math.floor(diff / (24 * 60 * 60 * 1000))}天前`
  } else {
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }
}

/**
 * 创建历史记录
 */
export const createHistoryRecord = (
  originalImage: string,
  options?: {
    resultImage?: string
    algorithm?: string
    algorithmId?: string
    fileName?: string
    status?: HistoryRecord['status']
    processingTime?: number
  }
): Omit<HistoryRecord, 'id'> => {
  return {
    originalImage,
    resultImage: options?.resultImage,
    algorithm: options?.algorithm,
    algorithmId: options?.algorithmId,
    fileName: options?.fileName,
    status: options?.status || 'success',
    processingTime: options?.processingTime,
    timestamp: new Date().toISOString(),
  }
}
