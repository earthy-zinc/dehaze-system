<template>
  <div class="hotword-panel">
    <div class="panel-tip">
      添加算法名、专业术语等热词，可提升语音识别对专业词汇的识别率
    </div>
    <div class="add-row">
      <el-input
        v-model="newWord"
        placeholder="输入热词，如：FFA-Net"
        maxlength="50"
        clearable
        @keyup.enter="handleAdd"
      />
      <el-button type="primary" :disabled="!newWord.trim()" @click="handleAdd">
        添加
      </el-button>
    </div>
    <HotwordList :words="voiceStore.hotwords" @delete="handleDelete" />
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import { useVoiceStore } from "@/store/modules/voice";
import HotwordList from "./HotwordList.vue";

defineOptions({ name: "HotwordManagePanel" });

const voiceStore = useVoiceStore();
const newWord = ref("");

async function handleAdd() {
  const word = newWord.value.trim();
  if (!word) return;
  try {
    // 热词数量超上限由后端校验并全局提示
    await voiceStore.addHotword(word);
    newWord.value = "";
    ElMessage.success("热词已添加");
  } catch {
    // 错误提示由全局拦截器处理
  }
}

async function handleDelete(id: number) {
  await voiceStore.deleteHotword(id);
  ElMessage.success("已删除");
}

onMounted(() => {
  voiceStore.fetchHotwords();
});
</script>

<style lang="scss" scoped>
.panel-tip {
  padding: 8px 0 12px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.add-row {
  display: flex;
  gap: 12px;
  padding-bottom: 8px;
}
</style>
