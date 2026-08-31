<template>
  <div class="preference-form" v-loading="loadingVoices">
    <div class="form-row">
      <div class="row-label">
        <span class="label-text">语音回复</span>
        <span class="label-desc">开启后 AI 对话回复自动语音播报</span>
      </div>
      <el-switch
        :model-value="voiceStore.ttsPreference.enabled"
        @change="
          (value: string | number | boolean) =>
            update({ enabled: value === true })
        "
      />
    </div>
    <div class="form-row">
      <div class="row-label">
        <span class="label-text">音色</span>
        <span class="label-desc">语音播报使用的声音</span>
      </div>
      <el-select
        :model-value="voiceStore.ttsPreference.voiceId"
        placeholder="默认音色"
        clearable
        style="width: 200px"
        @change="(value: string) => update({ voiceId: value ?? '' })"
      >
        <el-option
          v-for="voice in voices"
          :key="voice.id"
          :label="voice.name"
          :value="voice.id"
        />
      </el-select>
    </div>
    <div class="form-row">
      <div class="row-label">
        <span class="label-text">语速</span>
        <span class="label-desc">播报语速</span>
      </div>
      <el-radio-group
        :model-value="voiceStore.ttsPreference.speed"
        @change="
          (value: string | number | boolean | undefined) =>
            update({ speed: Number(value) })
        "
      >
        <el-radio-button :value="0.8">慢速 0.8x</el-radio-button>
        <el-radio-button :value="1">正常 1.0x</el-radio-button>
        <el-radio-button :value="1.2">快速 1.2x</el-radio-button>
      </el-radio-group>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import { VoiceAPI, type VoiceVO } from "dehaze-sdk-js";
import { useVoiceStore, type TtsPreference } from "@/store/modules/voice";

defineOptions({ name: "VoicePreferenceForm" });

const voiceStore = useVoiceStore();
const voices = ref<VoiceVO[]>([]);
const loadingVoices = ref(false);

function update(partial: Partial<TtsPreference>) {
  voiceStore.updatePreference(partial);
}

onMounted(async () => {
  loadingVoices.value = true;
  try {
    voices.value = await VoiceAPI.getVoices();
  } finally {
    loadingVoices.value = false;
  }
});
</script>

<style lang="scss" scoped>
.form-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);

  &:last-child {
    border-bottom: none;
  }

  .row-label {
    display: flex;
    flex-direction: column;
    gap: 2px;

    .label-text {
      font-size: 14px;
      font-weight: 500;
      color: var(--el-text-color-primary);
    }

    .label-desc {
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }
}
</style>
