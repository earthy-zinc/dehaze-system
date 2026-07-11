/**
 * 历史记录服务
 * 接入云端 SDK ImageInputHistoryAPI，对齐设计文档 API接口.md §2.3
 */

import { ImageInputHistoryAPI } from 'dehaze-sdk-js'
import type { InputHistoryVO, HistoryForm, HistoryQuery } from 'dehaze-sdk-js'

/** 分组后的历史记录 */
export interface GroupedHistory {
  title: string
  records: InputHistoryVO[]
}

/** 分页查询历史记录 */
export const getHistoryPage = async (query?: HistoryQuery): Promise<{ list: InputHistoryVO[]; total: number }> => {
  const res = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 100, ...query })
  // SDK PageResult<T> 的 list 类型定义为 T，但后端实际返回数组，此处强制转换
  const list = (res.list as unknown as InputHistoryVO[]) || []
  return { list, total: res.total || 0 }
}

/** 获取历史记录详情 */
export const getHistoryById = (id: number) => ImageInputHistoryAPI.getById(id)

/** 创建历史记录 */
export const createHistoryRecord = (data: HistoryForm) => ImageInputHistoryAPI.create(data)

/** 更新历史记录（如收藏） */
export const updateHistoryRecord = (id: number, data: { isFavorite?: boolean }) =>
  ImageInputHistoryAPI.update(id, data)

/** 删除单条历史记录 */
export const deleteHistoryRecord = (id: number) => ImageInputHistoryAPI.deleteById(id)

/** 清空所有历史记录 */
export const clearAllHistory = () => ImageInputHistoryAPI.clearAll()

/**
 * 将历史记录按时间分组
 */
export const groupHistoryByDate = (records: InputHistoryVO[]): GroupedHistory[] => {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000)
  const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)

  const groups: Record<string, InputHistoryVO[]> = {
    '今天': [],
    '昨天': [],
    '最近7天': [],
    '更早': [],
  }

  records.forEach(record => {
    const ts = record.createTime
    if (!ts) {
      groups['更早'].push(record)
      return
    }
    const recordDate = new Date(ts)
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

  return Object.entries(groups)
    .filter(([, recs]) => recs.length > 0)
    .map(([title, recs]) => ({ title, records: recs }))
}

/**
 * 格式化时间显示
 */
export const formatTimestamp = (timestamp?: string): string => {
  if (!timestamp) return ''
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
