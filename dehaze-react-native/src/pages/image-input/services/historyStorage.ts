/**
 * 历史记录本地存储服务
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { HistoryRecord, HistoryGroup } from '../types/imageInput';

const STORAGE_KEY = 'dehaze_history';
const MAX_RECORDS = 20;

export const historyStorage = {
  /**
   * 获取所有历史记录
   */
  getHistory: async (): Promise<HistoryRecord[]> => {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEY);
      if (data) {
        return JSON.parse(data) as HistoryRecord[];
      }
      return [];
    } catch (error) {
      console.error('Failed to get history:', error);
      return [];
    }
  },

  /**
   * 添加历史记录
   */
  addRecord: async (record: Omit<HistoryRecord, 'id' | 'timestamp'>): Promise<void> => {
    try {
      const history = await historyStorage.getHistory();

      const newRecord: HistoryRecord = {
        ...record,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
      };

      // 添加到开头
      history.unshift(newRecord);

      // 限制记录数量
      if (history.length > MAX_RECORDS) {
        history.splice(MAX_RECORDS);
      }

      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (error) {
      console.error('Failed to add history record:', error);
      throw error;
    }
  },

  /**
   * 删除单条记录
   */
  deleteRecord: async (id: string): Promise<void> => {
    try {
      const history = await historyStorage.getHistory();
      const filtered = history.filter(record => record.id !== id);
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    } catch (error) {
      console.error('Failed to delete history record:', error);
      throw error;
    }
  },

  /**
   * 清空所有历史记录
   */
  clearHistory: async (): Promise<void> => {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Failed to clear history:', error);
      throw error;
    }
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
      const recordDate = new Date(record.timestamp);

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

    // 过滤空分组并转换为数组
    return Object.entries(groups)
      .filter(([_, data]) => data.length > 0)
      .map(([title, data]) => ({ title, data }));
  },

  /**
   * 格式化时间显示
   */
  formatTimestamp: (timestamp: string): string => {
    const date = new Date(timestamp);
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

export { MAX_RECORDS };
