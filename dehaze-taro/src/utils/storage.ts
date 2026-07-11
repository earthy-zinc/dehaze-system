import Taro from '@tarojs/taro';
import { TOKEN_KEY } from '@/enums/CacheEnum';

const STORAGE_KEYS = {
  USER_INFO: 'userInfo',
  PERMISSIONS: 'permissions',
  ROLES: 'roles',
  SELECTED_DEPT: 'selectedDept',
  CACHE_EXPIRE: 'cacheExpire',
} as const;

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
        data: JSON.stringify(data)
      });
    } catch (error) {
      console.error('存储失败:', error);
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
      console.error('删除存储失败:', error);
    }
  }

  // 清空所有存储
  async clear(): Promise<void> {
    try {
      await Taro.clearStorage();
    } catch (error) {
      console.error('清空存储失败:', error);
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
      console.error('删除Token失败:', error);
    }
  }

  // 用户信息管理
  async setUserInfo(userInfo: any): Promise<void> {
    await this.setItem(STORAGE_KEYS.USER_INFO, userInfo);
  }

  async getUserInfo(): Promise<any | null> {
    return this.getItem(STORAGE_KEYS.USER_INFO);
  }

  // 权限管理
  async setPermissions(permissions: string[]): Promise<void> {
    await this.setItem(STORAGE_KEYS.PERMISSIONS, permissions);
  }

  async getPermissions(): Promise<string[]> {
    const permissions = await this.getItem<string[]>(STORAGE_KEYS.PERMISSIONS);
    return permissions || [];
  }

  // 角色管理
  async setRoles(roles: string[]): Promise<void> {
    await this.setItem(STORAGE_KEYS.ROLES, roles);
  }

  async getRoles(): Promise<string[]> {
    const roles = await this.getItem<string[]>(STORAGE_KEYS.ROLES);
    return roles || [];
  }

  // 部门管理
  async setSelectedDept(deptId: number | null): Promise<void> {
    if (deptId) {
      await this.setItem(STORAGE_KEYS.SELECTED_DEPT, deptId);
    } else {
      await this.removeItem(STORAGE_KEYS.SELECTED_DEPT);
    }
  }

  async getSelectedDept(): Promise<number | null> {
    return this.getItem(STORAGE_KEYS.SELECTED_DEPT);
  }
}

export const storage = new StorageManager();
