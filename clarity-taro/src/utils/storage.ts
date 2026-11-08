import Taro from "@tarojs/taro";
import { CacheEnum } from "@/enums/CacheEnum";
import { SESSION_KEY, type UserInfo } from "clarity-sdk-js";

// Taro 的 Sync 存储 API 在 RN 端没有实现（会弹「暂不支持」警告）。
// RN 改为「内存缓存 + 异步持久化」：读走缓存，写同时更新缓存并异步落盘。
// 冷启动第一次读可能为空，需要恢复登录态时由 preload 预载一次。
const isRN = process.env.TARO_ENV === "rn";
const syncCache = new Map<string, string | null>();

/** 启动时调用，把需要同步读取的 key 预热进内存 */
export async function preloadSyncKeys(keys: string[]): Promise<void> {
  if (!isRN) return;
  await Promise.all(
    keys.map(async (key) => {
      try {
        const result = await Taro.getStorage({ key });
        syncCache.set(key, (result.data as string) ?? null);
      } catch {
        syncCache.set(key, null);
      }
    })
  );
}

/** 通用同步读写：小程序 / H5 走 Taro，RN 走内存缓存 + 异步落盘 */
export const syncStorage = {
  get<T = string>(key: string): T | null {
    try {
      if (isRN) return (syncCache.get(key) ?? null) as T | null;
      return (Taro.getStorageSync(key) as T) || null;
    } catch {
      return null;
    }
  },
  set(key: string, value: string): void {
    if (isRN) {
      syncCache.set(key, value);
      void Taro.setStorage({ key, data: value });
      return;
    }
    Taro.setStorageSync(key, value);
  },
  remove(key: string): void {
    if (isRN) {
      syncCache.delete(key);
      void Taro.removeStorage({ key });
      return;
    }
    Taro.removeStorageSync(key);
  },
};

class StorageManager {
  async setItem<T>(key: string, value: T, expire?: number): Promise<void> {
    try {
      const data = {
        value,
        expire: expire ? Date.now() + expire : null,
      };

      await Taro.setStorage({
        key,
        data: JSON.stringify(data),
      });
    } catch (error) {
      console.error("存储失败:", error);
      throw error;
    }
  }

  async getItem<T>(key: string): Promise<T | null> {
    try {
      const result = await Taro.getStorage({ key });
      const { value, expire } = JSON.parse(result.data);

      if (expire && Date.now() > expire) {
        await this.removeItem(key);
        return null;
      }

      return value;
    } catch (error) {
      return null;
    }
  }

  async removeItem(key: string): Promise<void> {
    try {
      await Taro.removeStorage({ key });
    } catch (error) {
      console.error("删除存储失败:", error);
    }
  }

  async clear(): Promise<void> {
    try {
      await Taro.clearStorage();
    } catch (error) {
      console.error("清空存储失败:", error);
    }
  }

  setSessionId(sessionId: string): void {
    if (isRN) {
      syncCache.set(SESSION_KEY, sessionId);
      void Taro.setStorage({ key: SESSION_KEY, data: sessionId });
      return;
    }
    Taro.setStorageSync(SESSION_KEY, sessionId);
  }

  getSessionId(): string | null {
    try {
      if (isRN) return syncCache.get(SESSION_KEY) ?? null;
      return Taro.getStorageSync(SESSION_KEY) || null;
    } catch {
      return null;
    }
  }

  removeSessionId(): void {
    try {
      if (isRN) {
        syncCache.delete(SESSION_KEY);
        void Taro.removeStorage({ key: SESSION_KEY });
        return;
      }
      Taro.removeStorageSync(SESSION_KEY);
    } catch (error) {
      console.error("删除SessionId失败:", error);
    }
  }

  async setUserInfo(userInfo: UserInfo): Promise<void> {
    await this.setItem(CacheEnum.USER_INFO, userInfo);
  }

  async getUserInfo(): Promise<UserInfo | null> {
    return this.getItem<UserInfo>(CacheEnum.USER_INFO);
  }

  async setPerms(perms: string[]): Promise<void> {
    await this.setItem(CacheEnum.PERMS, perms);
  }

  async getPerms(): Promise<string[]> {
    const perms = await this.getItem<string[]>(CacheEnum.PERMS);
    return perms || [];
  }

  async setRoles(roles: string[]): Promise<void> {
    await this.setItem(CacheEnum.ROLES, roles);
  }

  async getRoles(): Promise<string[]> {
    const roles = await this.getItem<string[]>(CacheEnum.ROLES);
    return roles || [];
  }

  clearAuth(): void {
    if (isRN) {
      [
        SESSION_KEY,
        CacheEnum.USER_INFO,
        CacheEnum.PERMS,
        CacheEnum.ROLES,
      ].forEach((key) => {
        syncCache.delete(key);
        void Taro.removeStorage({ key });
      });
      return;
    }
    Taro.removeStorageSync(SESSION_KEY);
    Taro.removeStorageSync(CacheEnum.USER_INFO);
    Taro.removeStorageSync(CacheEnum.PERMS);
    Taro.removeStorageSync(CacheEnum.ROLES);
  }
}

export const storage = new StorageManager();
