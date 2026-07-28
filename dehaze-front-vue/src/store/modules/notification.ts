import { store } from "@/store";

import { MessageAPI } from "dehaze-sdk-js";

export const useNotificationStore = defineStore("notification", () => {
  const unreadCount = ref(0);
  const loaded = ref(false);

  async function fetchUnreadCount() {
    const data = await MessageAPI.getUnreadCount();
    unreadCount.value = data.count;
    loaded.value = true;
  }

  function increment() {
    unreadCount.value += 1;
  }

  function decrement() {
    if (unreadCount.value > 0) unreadCount.value -= 1;
  }

  function reset() {
    unreadCount.value = 0;
    loaded.value = false;
  }

  return {
    unreadCount,
    loaded,
    fetchUnreadCount,
    increment,
    decrement,
    reset,
  };
});

export function useNotificationStoreHook() {
  return useNotificationStore(store);
}
