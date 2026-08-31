<!-- 知识库配置表单（创建/编辑共用）：分块与向量化模型创建后不可修改 -->
<script lang="ts" setup>
import type {
  AiModelVO,
  ChunkingStrategy,
  KnowledgeBaseCreateForm,
} from "dehaze-sdk-js";
import { AiModelAPI } from "dehaze-sdk-js";
import type { FormItemRule } from "element-plus";
import { computed, onMounted, reactive, ref, watch } from "vue";

defineOptions({ name: "KbConfigForm" });

const props = defineProps<{
  modelValue: Partial<KnowledgeBaseCreateForm>;
  mode: "create" | "edit";
  scope: "self" | "admin";
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: Partial<KnowledgeBaseCreateForm>): void;
  (e: "submit", value: Partial<KnowledgeBaseCreateForm>): void;
}>();

// scope=self 推荐默认值：普通用户不感知底层参数，需在创建时定好不可改项
const RECOMMENDED_DEFAULTS = {
  visibility: "private",
  chunkingStrategy: "fixed",
  chunkSize: 800,
  chunkOverlap: 80,
  searchStrategy: "hybrid",
  topK: 5,
  scoreThreshold: 0.5,
} as const;

const CHUNKING_OPTIONS: { value: ChunkingStrategy; label: string }[] = [
  { value: "fixed", label: "固定长度" },
  { value: "semantic", label: "语义分块" },
  { value: "recursive", label: "递归分块" },
  { value: "qa", label: "问答对" },
  { value: "table", label: "表格解析" },
];

const SEARCH_OPTIONS: {
  value: NonNullable<KnowledgeBaseCreateForm["searchStrategy"]>;
  label: string;
}[] = [
  { value: "vector", label: "向量检索" },
  { value: "keyword", label: "关键词检索" },
  { value: "hybrid", label: "混合检索" },
];

const form = reactive<Partial<KnowledgeBaseCreateForm>>({
  ...props.modelValue,
});
if (props.mode === "create") {
  // 仅填充未提供的字段，保留父组件已预设的值
  if (form.visibility === undefined)
    form.visibility = RECOMMENDED_DEFAULTS.visibility;
  if (form.chunkingStrategy === undefined)
    form.chunkingStrategy = RECOMMENDED_DEFAULTS.chunkingStrategy;
  if (form.chunkSize === undefined)
    form.chunkSize = RECOMMENDED_DEFAULTS.chunkSize;
  if (form.chunkOverlap === undefined)
    form.chunkOverlap = RECOMMENDED_DEFAULTS.chunkOverlap;
  if (form.searchStrategy === undefined)
    form.searchStrategy = RECOMMENDED_DEFAULTS.searchStrategy;
  if (form.topK === undefined) form.topK = RECOMMENDED_DEFAULTS.topK;
  if (form.scoreThreshold === undefined)
    form.scoreThreshold = RECOMMENDED_DEFAULTS.scoreThreshold;
}

watch(form, () => emit("update:modelValue", { ...form }), { deep: true });

const formRef = ref();
const embeddingModels = ref<AiModelVO[]>([]);
const rerankModels = ref<AiModelVO[]>([]);

// 分块参数与向量化模型创建后不可修改（已有分块与 ES 索引基于其生成）
const immutableDisabled = computed(() => props.mode === "edit");

const rules = computed<Record<string, FormItemRule[]>>(() => {
  const base: Record<string, FormItemRule[]> = {
    name: [{ required: true, message: "请输入知识库名称", trigger: "blur" }],
  };
  if (props.mode === "create") {
    base.embeddingModel = [
      { required: true, message: "请选择向量化模型", trigger: "change" },
    ];
  }
  return base;
});

async function loadModelOptions() {
  try {
    const [embedding, rerank] = await Promise.all([
      AiModelAPI.listEnabledModels("embedding"),
      AiModelAPI.listEnabledModels("rerank"),
    ]);
    embeddingModels.value = embedding ?? [];
    rerankModels.value = rerank ?? [];
  } catch {
    // 模型注册表加载失败不阻塞表单，下拉为空即用户可感知
  }
}

onMounted(loadModelOptions);

function handleSubmit() {
  formRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    if (props.mode === "edit") {
      // 编辑仅提交可修改项，避免覆盖创建后不可变字段
      emit("submit", {
        name: form.name,
        description: form.description,
        searchStrategy: form.searchStrategy,
        hybridWeight: form.hybridWeight,
        topK: form.topK,
        scoreThreshold: form.scoreThreshold,
        enableRerank: form.enableRerank,
        rerankModel: form.enableRerank ? form.rerankModel : undefined,
      });
    } else {
      emit("submit", { ...form });
    }
  });
}
</script>

<template>
  <div>
    <el-alert
      v-if="mode === 'create' && scope === 'self'"
      type="info"
      show-icon
      :closable="false"
      class="mb-4"
      title="向量化模型、分块策略、分块大小、重叠在创建后不可修改"
    />
    <el-form ref="formRef" :model="form" :rules="rules" label-width="110px">
      <el-form-item label="名称" prop="name">
        <el-input
          v-model="form.name"
          placeholder="请输入知识库名称"
          maxlength="50"
        />
      </el-form-item>
      <el-form-item label="描述">
        <el-input
          v-model="form.description"
          type="textarea"
          :rows="2"
          placeholder="请输入知识库描述"
          maxlength="200"
        />
      </el-form-item>
      <el-form-item v-if="mode === 'create'" label="可见性">
        <el-radio-group v-model="form.visibility">
          <el-radio value="private">私有（仅自己可见）</el-radio>
          <el-radio value="public">公共（全员只读）</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="向量化模型" prop="embeddingModel">
        <el-select
          v-model="form.embeddingModel"
          :disabled="immutableDisabled"
          placeholder="请选择向量化模型"
          class="w-full"
        >
          <el-option
            v-for="m in embeddingModels"
            :key="m.id"
            :label="
              m.displayName ? `${m.displayName}（${m.modelId}）` : m.modelId
            "
            :value="m.modelId"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="分块策略">
        <el-select
          v-model="form.chunkingStrategy"
          :disabled="immutableDisabled"
          class="w-full"
        >
          <el-option
            v-for="o in CHUNKING_OPTIONS"
            :key="o.value"
            :label="o.label"
            :value="o.value"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="分块大小">
        <el-input-number
          v-model="form.chunkSize"
          :disabled="immutableDisabled"
          :min="100"
          :max="4000"
          :step="50"
        />
        <span class="form-tip">token</span>
      </el-form-item>
      <el-form-item label="分块重叠">
        <el-input-number
          v-model="form.chunkOverlap"
          :disabled="immutableDisabled"
          :min="0"
          :max="1000"
          :step="10"
        />
        <span class="form-tip">token</span>
      </el-form-item>
      <el-form-item label="检索策略">
        <el-select v-model="form.searchStrategy" class="w-full">
          <el-option
            v-for="o in SEARCH_OPTIONS"
            :key="o.value"
            :label="o.label"
            :value="o.value"
          />
        </el-select>
      </el-form-item>
      <el-form-item v-if="form.searchStrategy === 'hybrid'" label="向量权重">
        <el-slider
          v-model="form.hybridWeight"
          :min="0"
          :max="1"
          :step="0.05"
          class="w-full"
        />
      </el-form-item>
      <el-form-item label="Top-K">
        <el-input-number v-model="form.topK" :min="1" :max="50" />
      </el-form-item>
      <el-form-item label="相似度阈值">
        <el-input-number
          v-model="form.scoreThreshold"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
        />
      </el-form-item>
      <el-form-item label="Rerank">
        <el-switch v-model="form.enableRerank" />
      </el-form-item>
      <el-form-item v-if="form.enableRerank" label="重排序模型">
        <el-select
          v-model="form.rerankModel"
          placeholder="请选择重排序模型"
          clearable
          class="w-full"
        >
          <el-option
            v-for="m in rerankModels"
            :key="m.id"
            :label="
              m.displayName ? `${m.displayName}（${m.modelId}）` : m.modelId
            "
            :value="m.modelId"
          />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="handleSubmit">
          {{ mode === "create" ? "创建" : "保存" }}
        </el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<style lang="scss" scoped>
.form-tip {
  margin-left: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
