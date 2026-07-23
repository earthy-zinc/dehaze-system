import Taro from "@tarojs/taro";
import { CacheEnum } from "@/enums/CacheEnum";
import { TOKEN_KEY, type UserInfo } from "dehaze-sdk-js";

// 存储管理类
class StorageManager {
  // 设置存储项
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

  // 获取存储项
  async getItem<T>(key: string): Promise<T | null> {
    try {
      const result = await Taro.getStorage({ key });
      const { value, expire } = JSON.parse(result.data);

      // 检查是否过期
      if (expire && Date.now() > expire) {
        await this.removeItem(key);
        return null;
      }

      return value;
    } catch (error) {
      return null;
    }
  }

  // 删除存储项
  async removeItem(key: string): Promise<void> {
    try {
      await Taro.removeStorage({ key });
    } catch (error) {
      console.error("删除存储失败:", error);
    }
  }

  // 清空所有存储
  async clear(): Promise<void> {
    try {
      await Taro.clearStorage();
    } catch (error) {
      console.error("清空存储失败:", error);
    }
  }

  // ===== Token 管理（同步存储，原始字符串，供 SDK 拦截器同步读取） =====

  setToken(token: string): void {
    Taro.setStorageSync(TOKEN_KEY, token);
  }

  getToken(): string | null {
    try {
      return Taro.getStorageSync(TOKEN_KEY) || null;
    } catch {
      return null;
    }
  }

  removeToken(): void {
    try {
      Taro.removeStorageSync(TOKEN_KEY);
    } catch (error) {
      console.error("删除Token失败:", error);
    }
  }

  // 用户信息管理
  async setUserInfo(userInfo: UserInfo): Promise<void> {
    await this.setItem(CacheEnum.USER_INFO, userInfo);
  }

  async getUserInfo(): Promise<UserInfo | null> {
    return this.getItem<UserInfo>(CacheEnum.USER_INFO);
  }

  // 权限管理
  async setPerms(perms: string[]): Promise<void> {
    await this.setItem(CacheEnum.PERMS, perms);
  }

  async getPerms(): Promise<string[]> {
    const perms = await this.getItem<string[]>(CacheEnum.PERMS);
    return perms || [];
  }

  // 角色管理
  async setRoles(roles: string[]): Promise<void> {
    await this.setItem(CacheEnum.ROLES, roles);
  }

  async getRoles(): Promise<string[]> {
    const roles = await this.getItem<string[]>(CacheEnum.ROLES);
    return roles || [];
  }

  /**
   * 同步清空本地认证信息（token + 用户信息 + 权限 + 角色）
   * 用于登出/登录失效跳转前，确保下次进入为干净状态
   */
  clearAuth(): void {
    Taro.removeStorageSync(TOKEN_KEY);
    Taro.removeStorageSync(CacheEnum.USER_INFO);
    Taro.removeStorageSync(CacheEnum.PERMS);
    Taro.removeStorageSync(CacheEnum.ROLES);
  }
}

export const storage = new StorageManager();
