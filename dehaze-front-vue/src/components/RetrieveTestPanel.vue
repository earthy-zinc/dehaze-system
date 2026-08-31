<!-- 检索测试面板：与对话共用检索链路，验证知识库召回效果 -->
<script lang="ts" setup>
import type { SearchResultVO } from "dehaze-sdk-js";
import { AiKnowledgeBaseAPI } from "dehaze-sdk-js";
import { ElMessage } from "element-plus";
import { ref } from "vue";

defineOptions({ name: "RetrieveTestPanel" });

const props = defineProps<{
  knowledgeBaseId: number;
  readonly?: boolean;
}>();

const question = ref("");
const loading = ref(false);
const result = ref<SearchResultVO | null>(null);

async function handleTest() {
  if (!question.value.trim()) {
    ElMessage.warning("请输入测试问题");
    return;
  }
  loading.value = true;
  try {
    result.value = await AiKnowledgeBaseAPI.retrieveTest(
      props.knowledgeBaseId,
      { query: question.value.trim() }
    );
  } catch {
    // 错误已由全局拦截器提示
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div>
    <div class="flex gap-2 mb-4">
      <el-input
        v-model="question"
        placeholder="输入测试问题，验证知识库召回效果"
        clearable
        :disabled="readonly"
        @keyup.enter="handleTest"
      />
      <el-tooltip v-if="readonly" content="公共知识库为只读，仅支持浏览">
        <el-button type="primary" disabled>检索测试</el-button>
      </el-tooltip>
      <el-button v-else type="primary" :loading="loading" @click="handleTest">
        检索测试
      </el-button>
    </div>

    <template v-if="result">
      <el-empty
        v-if="result.results.length === 0"
        description="未召回相关片段"
        :image-size="80"
      />
      <div
        v-for="item in result.results"
        :key="item.chunkId"
        class="result-item"
      >
        <div class="flex items-center justify-between mb-1">
          <span class="text-sm font-bold">
            {{ item.documentTitle }}
            <span class="ml-1 font-normal text-gray-400">
              分块 #{{ item.chunkIndex }}
            </span>
          </span>
          <el-tag type="primary" size="small">
            相似度 {{ (item.score * 100).toFixed(1) }}%
          </el-tag>
        </div>
        <div class="result-content">{{ item.content }}</div>
      </div>
    </template>
  </div>
</template>

<style lang="scss" scoped>
.result-item {
  padding: 12px;
  margin-bottom: 12px;
  background: var(--el-fill-color-lighter);
  border-radius: 6px;
}

.result-content {
  display: -webkit-box;
  overflow: hidden;
  -webkit-line-clamp: 5;
  font-size: 13px;
  line-height: 1.6;
  color: var(--el-text-color-regular);
  word-break: break-all;
  white-space: pre-wrap;
  -webkit-box-orient: vertical;
}
</style>
