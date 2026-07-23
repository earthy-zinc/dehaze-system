/**
 * AsyncStorage 封装
 *
 * 统一使用 JSON 序列化存储，避免 string 与对象存储格式不一致。
 */
import AsyncStorage from '@react-native-async-storage/async-storage';

export const storage = {
  async get<T>(key: string): Promise<T | null> {
    const value = await AsyncStorage.getItem(key);
    if (value == null) {
      return null;
    }
    return JSON.parse(value) as T;
  },

  async set(key: string, value: unknown): Promise<void> {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  },

  async remove(key: string): Promise<void> {
    await AsyncStorage.removeItem(key);
  },
};
