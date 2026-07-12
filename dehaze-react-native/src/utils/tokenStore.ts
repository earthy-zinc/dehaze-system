/**
 * Token 内存存储
 *
 * SDK 的 getToken 回调需要同步返回 token，
 * 而 AsyncStorage 是异步的，因此维护一份内存副本，
 * 由 AuthContext 在登录/恢复时同步写入。
 */

let accessToken: string | null = null;
let onTokenInvalid: (() => void) | null = null;

export const tokenStore = {
  /** 同步获取 token */
  get(): string | null {
    return accessToken;
  },
  /** 设置 token（登录成功或从 AsyncStorage 恢复时调用） */
  set(token: string) {
    accessToken = token;
  },
  /** 清空 token（注销或 token 失效时调用） */
  clear() {
    accessToken = null;
  },
};

export function setOnTokenInvalid(cb: (() => void) | null) {
  onTokenInvalid = cb;
}

export function triggerTokenInvalid() {
  onTokenInvalid?.();
}
