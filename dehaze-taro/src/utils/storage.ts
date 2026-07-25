import Taro from "@tarojs/taro";
import { CacheEnum } from "@/enums/CacheEnum";
import { SESSION_KEY, type UserInfo } from "dehaze-sdk-js";

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
    Taro.setStorageSync(SESSION_KEY, sessionId);
  }

  getSessionId(): string | null {
    try {
      return Taro.getStorageSync(SESSION_KEY) || null;
    } catch {
      return null;
    }
  }

  removeSessionId(): void {
    try {
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
    Taro.removeStorageSync(SESSION_KEY);
    Taro.removeStorageSync(CacheEnum.USER_INFO);
    Taro.removeStorageSync(CacheEnum.PERMS);
    Taro.removeStorageSync(CacheEnum.ROLES);
  }
}

export const storage = new StorageManager();
