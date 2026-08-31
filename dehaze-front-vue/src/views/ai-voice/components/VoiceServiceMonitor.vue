<!-- 语音服务监控：ASR/TTS 引擎在线状态、并发会话数、模型加载状态，30s 轮询 -->
<script lang="ts" setup>
import { onMounted, onUnmounted } from "vue";
import { Refresh } from "@element-plus/icons-vue";
import { useAdminVoiceStore } from "@/store/modules/adminVoice";

const adminVoiceStore = useAdminVoiceStore();

function engineTagType(status: "online" | "offline") {
  return status === "online" ? "success" : "danger";
}

onMounted(() => {
  adminVoiceStore.startMonitor();
});

onUnmounted(() => {
  adminVoiceStore.stopMonitor();
});
</script>

<template>
  <div v-loading="adminVoiceStore.statusLoading">
    <div class="flex justify-end mb-4">
      <el-button @click="adminVoiceStore.fetchServiceStatus()">
        <el-icon><Refresh /></el-icon>
        立即刷新
      </el-button>
    </div>

    <el-empty
      v-if="!adminVoiceStore.serviceStatus"
      description="暂无状态数据"
    />

    <template v-else>
      <el-row :gutter="16">
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <div class="flex justify-between items-center">
                <span class="font-bold">语音识别（ASR）</span>
                <el-tag
                  :type="
                    engineTagType(
                      adminVoiceStore.serviceStatus.asr.engineStatus
                    )
                  "
                >
                  {{
                    adminVoiceStore.serviceStatus.asr.engineStatus === "online"
                      ? "在线"
                      : "离线"
                  }}
                </el-tag>
              </div>
            </template>
            <el-descriptions :column="1" border>
              <el-descriptions-item label="并发会话数">
                {{ adminVoiceStore.serviceStatus.asr.concurrentSessions }} /
                {{ adminVoiceStore.serviceStatus.asr.maxConcurrentSessions }}
              </el-descriptions-item>
              <el-descriptions-item label="流式模型">
                <el-tag
                  :type="
                    adminVoiceStore.serviceStatus.asr.streamModelLoaded
                      ? 'success'
                      : 'info'
                  "
                >
                  {{
                    adminVoiceStore.serviceStatus.asr.streamModelLoaded
                      ? "已加载"
                      : "未加载"
                  }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="离线模型">
                <el-tag
                  :type="
                    adminVoiceStore.serviceStatus.asr.offlineModelLoaded
                      ? 'success'
                      : 'info'
                  "
                >
                  {{
                    adminVoiceStore.serviceStatus.asr.offlineModelLoaded
                      ? "已加载"
                      : "未加载"
                  }}
                </el-tag>
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </el-col>

        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <div class="flex justify-between items-center">
                <span class="font-bold">语音合成（TTS）</span>
                <el-tag
                  :type="
                    engineTagType(
                      adminVoiceStore.serviceStatus.tts.engineStatus
                    )
                  "
                >
                  {{
                    adminVoiceStore.serviceStatus.tts.engineStatus === "online"
                      ? "在线"
                      : "离线"
                  }}
                </el-tag>
              </div>
            </template>
            <el-descriptions :column="1" border>
              <el-descriptions-item label="音色模型">
                <el-tag
                  :type="
                    adminVoiceStore.serviceStatus.tts.voiceModelLoaded
                      ? 'success'
                      : 'info'
                  "
                >
                  {{
                    adminVoiceStore.serviceStatus.tts.voiceModelLoaded
                      ? "已加载"
                      : "未加载"
                  }}
                </el-tag>
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </el-col>
      </el-row>
    </template>
  </div>
</template>
