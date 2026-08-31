import { HotwordVO, ServiceStatusVO, VoiceAPI } from "dehaze-sdk-js";

// 管理端语音面板 Store：全局热词管理 + 语音服务状态监控（30s 轮询）
export const useAdminVoiceStore = defineStore("adminVoice", () => {
  const monitorTab = ref<"hotword" | "service" | "engine">("hotword");

  const globalHotwords = ref<HotwordVO[]>([]);
  const hotwordLoading = ref(false);

  const serviceStatus = ref<ServiceStatusVO | null>(null);
  const statusLoading = ref(false);
  let monitorTimer: ReturnType<typeof setInterval> | null = null;

  async function fetchGlobalHotwords() {
    hotwordLoading.value = true;
    try {
      globalHotwords.value = await VoiceAPI.getGlobalHotwords();
    } finally {
      hotwordLoading.value = false;
    }
  }

  async function addGlobalHotword(word: string) {
    await VoiceAPI.addGlobalHotword({ word });
    ElMessage.success("全局热词已添加");
    await fetchGlobalHotwords();
  }

  async function deleteGlobalHotword(id: number) {
    await VoiceAPI.deleteGlobalHotword(id);
    ElMessage.success("全局热词已删除");
    await fetchGlobalHotwords();
  }

  async function fetchServiceStatus() {
    statusLoading.value = true;
    try {
      serviceStatus.value = await VoiceAPI.getServiceStatus();
    } finally {
      statusLoading.value = false;
    }
  }

  function startMonitor() {
    if (monitorTimer) return;
    fetchServiceStatus();
    monitorTimer = setInterval(fetchServiceStatus, 30 * 1000);
  }

  function stopMonitor() {
    if (monitorTimer) {
      clearInterval(monitorTimer);
      monitorTimer = null;
    }
  }

  return {
    monitorTab,
    globalHotwords,
    hotwordLoading,
    serviceStatus,
    statusLoading,
    fetchGlobalHotwords,
    addGlobalHotword,
    deleteGlobalHotword,
    fetchServiceStatus,
    startMonitor,
    stopMonitor,
  };
});
