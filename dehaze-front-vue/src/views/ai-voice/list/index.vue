<!-- 管理端语音面板：全局热词配置 + 服务监控 + 引擎配置，按权限控制各 tab 显示 -->
<script lang="ts" setup>
import { useAdminVoiceStore } from "@/store/modules/adminVoice";
import { useUserStore } from "@/store/modules/user";
import GlobalHotwordPanel from "../components/GlobalHotwordPanel.vue";
import VoiceServiceMonitor from "../components/VoiceServiceMonitor.vue";
import VoiceEnginePanel from "../components/VoiceEnginePanel.vue";

defineOptions({ name: "AiVoiceList" });

const adminVoiceStore = useAdminVoiceStore();
const userStore = useUserStore();

const hasPerm = (perm: string) => {
  const { roles, perms } = userStore.user;
  return roles.includes("ROOT") || (perms?.includes(perm) ?? false);
};
</script>

<template>
  <div class="app-container">
    <el-tabs v-model="adminVoiceStore.monitorTab" class="bg-white p-4 rounded">
      <el-tab-pane
        v-if="hasPerm('voice:hotword:edit')"
        label="全局热词"
        name="hotword"
      >
        <GlobalHotwordPanel />
      </el-tab-pane>
      <el-tab-pane
        v-if="hasPerm('voice:service:monitor')"
        label="服务监控"
        name="service"
      >
        <VoiceServiceMonitor />
      </el-tab-pane>
      <el-tab-pane
        v-if="hasPerm('voice:engine:manage')"
        label="引擎配置"
        name="engine"
      >
        <VoiceEnginePanel />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>
