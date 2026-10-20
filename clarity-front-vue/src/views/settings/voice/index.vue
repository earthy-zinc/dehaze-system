<template>
  <div class="app-container voice-settings">
    <div class="page-header">
      <div class="header-title">
        <el-button link @click="router.back()">
          <el-icon><ArrowLeft /></el-icon>
        </el-button>
        <span class="title-text">语音设置</span>
      </div>
    </div>

    <section class="settings-section">
      <div class="section-header">
        <div class="section-title">
          <el-icon class="section-icon"><Bell /></el-icon>
          <span>播报偏好</span>
        </div>
        <span class="section-desc">控制 AI 回复的语音播报方式，仅本人生效</span>
      </div>
      <div class="section-body">
        <VoicePreferenceForm />
      </div>
    </section>

    <section class="settings-section">
      <div class="section-header">
        <div class="section-title">
          <el-icon class="section-icon"><Collection /></el-icon>
          <span>用户热词</span>
        </div>
        <span class="section-desc"
          >添加领域术语提升语音识别对专业词汇的准确率</span
        >
      </div>
      <div class="section-body">
        <HotwordManagePanel />
      </div>
    </section>
  </div>
</template>

<script lang="ts" setup>
import { useRouter } from "vue-router";
import { ArrowLeft, Bell, Collection } from "@element-plus/icons-vue";
import VoicePreferenceForm from "./components/VoicePreferenceForm.vue";
import HotwordManagePanel from "./components/HotwordManagePanel.vue";

defineOptions({ name: "SettingsVoice" });

const router = useRouter();
</script>

<style lang="scss" scoped>
.voice-settings {
  max-width: 720px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  margin-bottom: 24px;

  .header-title {
    display: flex;
    gap: 8px;
    align-items: center;

    .title-text {
      font-size: 22px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }
  }
}

.settings-section {
  margin-bottom: 20px;
  overflow: hidden;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;

  .section-header {
    padding: 16px 20px;
    background: var(--el-fill-color-lighter);
    border-bottom: 1px solid var(--el-border-color-lighter);

    .section-title {
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 15px;
      font-weight: 600;
      color: var(--el-text-color-primary);

      .section-icon {
        font-size: 18px;
        color: var(--el-color-primary);
      }
    }

    .section-desc {
      display: block;
      margin-top: 4px;
      margin-left: 26px;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .section-body {
    padding: 8px 20px;
  }
}
</style>
