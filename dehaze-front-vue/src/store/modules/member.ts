import { store } from "@/store";

import { MemberAPI, MemberProfileVO } from "dehaze-sdk-js";

export const useMemberStore = defineStore("member", () => {
  const profile = ref<MemberProfileVO>();
  const loading = ref(false);

  // 进行中的加载 Promise，用于合并并发请求，完成后置空以便重新拉取
  let pending: Promise<MemberProfileVO> | null = null;

  function requestProfile(): Promise<MemberProfileVO> {
    if (!pending) {
      loading.value = true;
      pending = MemberAPI.getProfile()
        .then((data) => {
          profile.value = data;
          return data;
        })
        .finally(() => {
          pending = null;
          loading.value = false;
        });
    }
    return pending;
  }

  /** 读取会员档案，已加载时直接复用缓存 */
  async function loadProfile(): Promise<MemberProfileVO> {
    if (profile.value) {
      return profile.value;
    }
    return requestProfile();
  }

  /** 强制重新加载会员档案（会员信息可能已变更的场景） */
  function refreshProfile(): Promise<MemberProfileVO> {
    return requestProfile();
  }

  /** 会员信息变更（购买成功/等级调整等）后失效缓存 */
  function invalidate() {
    profile.value = undefined;
    pending = null;
  }

  return {
    profile,
    loading,
    loadProfile,
    refreshProfile,
    invalidate,
  };
});

export function useMemberStoreHook() {
  return useMemberStore(store);
}
