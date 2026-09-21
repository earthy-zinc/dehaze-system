import { store } from "@/store";

import { DictAPI, OptionType } from "dehaze-sdk-js";

export const useDictStore = defineStore("dict", () => {
  // 按 typeCode 缓存选项请求，同会话内不重复请求；值为 Promise 以合并并发请求
  const dictCache = ref(new Map<string, Promise<OptionType[]>>());
  const loading = ref(false);

  async function getOptions(typeCode: string): Promise<OptionType[]> {
    let pending = dictCache.value.get(typeCode);
    if (!pending) {
      loading.value = true;
      pending = DictAPI.getDictOptions(typeCode).catch((error) => {
        dictCache.value.delete(typeCode);
        throw error;
      });
      dictCache.value.set(typeCode, pending);
      try {
        await pending;
      } finally {
        loading.value = false;
      }
    }
    return pending;
  }

  function invalidate(typeCode?: string) {
    if (typeCode) {
      dictCache.value.delete(typeCode);
    } else {
      dictCache.value.clear();
    }
  }

  function preload(typeCodes: string[]) {
    return Promise.all(typeCodes.map((code) => getOptions(code)));
  }

  return {
    dictCache,
    loading,
    getOptions,
    invalidate,
    preload,
  };
});

export function useDictStoreHook() {
  return useDictStore(store);
}
