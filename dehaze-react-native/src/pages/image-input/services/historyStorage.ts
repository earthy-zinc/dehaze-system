/**
 * 历史记录服务
 *
 * 通过 SDK 的 ImageInputHistoryAPI 对接后端 /api/v1/image-input/history。
 */

import { ImageInputHistoryAPI } from 'dehaze-sdk-js';
import type { HistoryForm } from 'dehaze-sdk-js';
import { HistoryRecord, HistoryGroup } from '../types/imageInput';

export const historyStorage = {
  /**
   * 获取历史记录列表（分页查询，默认取前 50 条）
   */
  getHistory: async (): Promise<HistoryRecord[]> => {
    const result = await ImageInputHistoryAPI.getPage({
      pageNum: 1,
      pageSize: 50,
    });
    return result.list;
  },

  /**
   * 添加历史记录
   */
  addRecord: async (record: HistoryForm): Promise<number> => {
    return await ImageInputHistoryAPI.create(record);
  },

  /**
   * 删除单条记录
   */
  deleteRecord: async (id: number): Promise<void> => {
    await ImageInputHistoryAPI.deleteById(id);
  },

  /**
   * 清空所有历史记录
   */
  clearHistory: async (confirm: boolean): Promise<void> => {
    await ImageInputHistoryAPI.clearAll(confirm);
  },

  /**
   * 将历史记录按时间分组
   */
  groupHistoryByDate: (history: HistoryRecord[]): HistoryGroup[] => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
    const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

    const groups: { [key: string]: HistoryRecord[] } = {
      '今天': [],
      '昨天': [],
      '最近7天': [],
      '更早': [],
    };

    history.forEach(record => {
      const recordDate = new Date(record.createTime || '');
      if (isNaN(recordDate.getTime())) return;

      if (recordDate >= today) {
        groups['今天'].push(record);
      } else if (recordDate >= yesterday) {
        groups['昨天'].push(record);
      } else if (recordDate >= lastWeek) {
        groups['最近7天'].push(record);
      } else {
        groups['更早'].push(record);
      }
    });

    return Object.entries(groups)
      .filter(([_, data]) => data.length > 0)
      .map(([title, data]) => ({ title, data }));
  },

  /**
   * 格式化时间显示
   */
  formatTimestamp: (timestamp?: string): string => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return '';
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 1) {
      return '刚刚';
    } else if (minutes < 60) {
      return `${minutes}分钟前`;
    } else if (hours < 24) {
      return `${hours}小时前`;
    } else if (days < 7) {
      return `${days}天前`;
    } else {
      return date.toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
      });
    }
  },
};
