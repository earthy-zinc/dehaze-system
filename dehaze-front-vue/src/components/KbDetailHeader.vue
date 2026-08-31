<!-- 知识库详情基本信息区：配置摘要 + 统计 -->
<script lang="ts" setup>
import type { KnowledgeBaseVO } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "KbDetailHeader" });

const props = defineProps<{
  kb: KnowledgeBaseVO;
}>();

const CHUNKING_LABELS: Record<string, string> = {
  fixed: "固定长度",
  semantic: "语义分块",
  recursive: "递归分块",
  qa: "问答对",
  table: "表格解析",
};

const SEARCH_LABELS: Record<string, string> = {
  vector: "向量检索",
  keyword: "关键词检索",
  hybrid: "混合检索",
};

const searchStrategyText = computed(() => {
  if (props.kb.searchStrategy === "hybrid") {
    return `混合检索（向量权重 ${props.kb.hybridWeight}）`;
  }
  return SEARCH_LABELS[props.kb.searchStrategy] ?? props.kb.searchStrategy;
});

const rerankText = computed(() =>
  props.kb.enableRerank === 1
    ? `开启（${props.kb.rerankModel ?? "-"}）`
    : "关闭"
);
</script>

<template>
  <el-descriptions :column="3" border>
    <el-descriptions-item label="名称" :span="3">
      <span class="font-bold">{{ kb.name }}</span>
      <el-tag
        :type="kb.visibility === 'public' ? 'success' : 'info'"
        size="small"
        class="ml-2"
      >
        {{ kb.visibility === "public" ? "公共" : "私有" }}
      </el-tag>
    </el-descriptions-item>
    <el-descriptions-item label="描述" :span="3">
      {{ kb.description || "-" }}
    </el-descriptions-item>
    <el-descriptions-item label="向量化模型">
      {{ kb.embeddingModel }}
    </el-descriptions-item>
    <el-descriptions-item label="分块策略">
      {{ CHUNKING_LABELS[kb.chunkingStrategy] ?? kb.chunkingStrategy }}
    </el-descriptions-item>
    <el-descriptions-item label="分块大小 / 重叠">
      {{ kb.chunkSize }} / {{ kb.chunkOverlap }} token
    </el-descriptions-item>
    <el-descriptions-item label="检索策略" :span="2">
      {{ searchStrategyText }}
    </el-descriptions-item>
    <el-descriptions-item label="Rerank">
      {{ rerankText }}
    </el-descriptions-item>
    <el-descriptions-item label="Top-K">
      {{ kb.topK }}
    </el-descriptions-item>
    <el-descriptions-item label="相似度阈值">
      {{ kb.scoreThreshold }}
    </el-descriptions-item>
    <el-descriptions-item label="创建时间">
      {{ kb.createTime ?? "-" }}
    </el-descriptions-item>
    <el-descriptions-item label="统计" :span="3">
      文档 {{ kb.documentCount }} · 分块 {{ kb.chunkCount }} · Token
      {{ kb.totalTokens }}
    </el-descriptions-item>
  </el-descriptions>
</template>
