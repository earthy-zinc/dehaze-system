<!-- AI 模型管理端页：模型注册表 + 价格版本 + 运营视图 -->
<template>
  <div class="app-container">
    <!-- 运营视图：用量/健康/降级统计 -->
    <el-card shadow="never" class="mb-[12px]">
      <template #header>
        <div class="flex justify-between items-center">
          <span>运营视图</span>
          <el-button
            :loading="modelStore.operationLoading"
            @click="modelStore.fetchOperation()"
          >
            <el-icon><Refresh /></el-icon>刷新
          </el-button>
        </div>
      </template>
      <el-tabs v-model="modelStore.statsTab">
        <el-tab-pane label="供应商健康" name="health">
          <el-table
            v-if="operation?.providerHealth?.length"
            :data="operation.providerHealth"
            size="small"
          >
            <el-table-column
              label="供应商"
              prop="providerName"
              min-width="120"
            />
            <el-table-column label="健康状态" width="90" align="center">
              <template #default="{ row }">
                <el-tag :type="providerHealthTag(row.health).type" size="small">
                  {{ providerHealthTag(row.health).label }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column
              label="调用量"
              prop="callCount"
              width="90"
              align="center"
            />
            <el-table-column label="成功率" width="90" align="center">
              <template #default="{ row }">{{ row.successRate }}%</template>
            </el-table-column>
            <el-table-column
              label="限流(429)"
              prop="rate429"
              width="100"
              align="center"
            />
            <el-table-column label="P95延迟" width="100" align="center">
              <template #default="{ row }">
                {{ row.p95LatencyMs != null ? row.p95LatencyMs + "ms" : "-" }}
              </template>
            </el-table-column>
            <el-table-column label="熔断" width="80" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.circuitOpen" type="danger" size="small"
                  >熔断中</el-tag
                >
                <span v-else>-</span>
              </template>
            </el-table-column>
          </el-table>
          <el-empty v-else description="暂无供应商健康数据" :image-size="60" />
        </el-tab-pane>
        <el-tab-pane label="模型用量分布" name="usage" lazy>
          <el-table
            v-if="operation?.modelUsage?.length"
            :data="operation.modelUsage"
            size="small"
          >
            <el-table-column label="模型" prop="displayName" min-width="140" />
            <el-table-column label="标识" prop="modelId" min-width="140" />
            <el-table-column
              label="调用数"
              prop="callCount"
              width="100"
              align="center"
            />
            <el-table-column
              label="输入Token"
              prop="inputTokens"
              width="110"
              align="center"
            />
            <el-table-column
              label="输出Token"
              prop="outputTokens"
              width="110"
              align="center"
            />
            <el-table-column
              label="积分开销"
              prop="credits"
              width="100"
              align="center"
            />
          </el-table>
          <el-empty v-else description="暂无用量数据" :image-size="60" />
        </el-tab-pane>
        <el-tab-pane label="降级与故障" name="degrade" lazy>
          <template v-if="operation?.degradeFault">
            <div class="mb-2 flex gap-6 text-sm">
              <span
                >Key 失败切换：<b>{{
                  operation.degradeFault.keyFailoverCount
                }}</b>
                次</span
              >
              <span
                >故障次数：<b>{{ operation.degradeFault.faultCount }}</b>
                次</span
              >
            </div>
            <el-table
              v-if="operation.degradeFault.downgradeFrequency?.length"
              :data="operation.degradeFault.downgradeFrequency"
              size="small"
            >
              <el-table-column label="模型" prop="modelId" min-width="180" />
              <el-table-column
                label="降级次数"
                prop="count"
                width="120"
                align="center"
              />
            </el-table>
            <el-empty v-else description="暂无降级记录" :image-size="60" />
          </template>
          <el-empty v-else description="暂无降级与故障数据" :image-size="60" />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 模型列表 -->
    <el-card shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-2">
            <el-input
              v-model="modelStore.query.keyword"
              clearable
              placeholder="名称/模型标识"
              style="width: 180px"
              @keyup.enter="handleQuery"
              @input="debouncedQuery"
            />
            <el-select
              v-model="modelStore.modelTypeFilter"
              class="!w-[120px]"
              @change="handleQuery"
            >
              <el-option label="全部类型" value="all" />
              <el-option label="对话" value="chat" />
              <el-option label="向量" value="embedding" />
              <el-option label="重排" value="rerank" />
            </el-select>
          </div>
          <el-button
            v-hasPerm="['ai:model:manage']"
            type="success"
            @click="openCreate()"
          >
            <el-icon><Plus /></el-icon>新增模型
          </el-button>
        </div>
      </template>

      <el-table v-loading="modelStore.loading" :data="modelStore.models">
        <el-table-column label="模型名称" min-width="150">
          <template #default="{ row }">
            <div>{{ row.displayName }}</div>
            <div class="text-xs text-gray-400">{{ row.modelId }}</div>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="80" align="center">
          <template #default="{ row }">
            <el-tag
              :type="typeTag[row.modelType as AiModelType].type"
              size="small"
            >
              {{ typeTag[row.modelType as AiModelType].label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="供应商" width="120">
          <template #default="{ row }">
            {{
              modelStore.providerNameMap.get(row.providerId) ?? row.providerId
            }}
          </template>
        </el-table-column>
        <el-table-column label="上下文" width="90" align="center">
          <template #default="{ row }">{{ row.maxContextTokens }}</template>
        </el-table-column>
        <el-table-column label="速度档位" width="90" align="center">
          <template #default="{ row }">
            {{
              row.speedTier ? (speedLabel[row.speedTier] ?? row.speedTier) : "-"
            }}
          </template>
        </el-table-column>
        <el-table-column label="降级" width="70" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.fallbackModelId" type="warning" size="small"
              >有</el-tag
            >
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="VIP等级" width="100" align="center">
          <template #default="{ row }">{{
            vipLabel[row.vipLevel] ?? row.vipLevel
          }}</template>
        </el-table-column>
        <el-table-column label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-switch
              v-model="row.status"
              :active-value="1"
              :inactive-value="0"
              @change="handleStatusChange(row as AiModelVO)"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              v-hasPerm="['ai:model:manage']"
              link
              type="primary"
              size="small"
              @click="openEdit(row as AiModelVO)"
            >
              编辑
            </el-button>
            <el-button
              v-hasPerm="['ai:model:manage']"
              link
              type="primary"
              size="small"
              @click="modelStore.openPriceDialog(row as AiModelVO)"
            >
              价格
            </el-button>
            <el-button
              v-hasPerm="['ai:model:manage']"
              link
              type="danger"
              size="small"
              @click="handleDelete(row as AiModelVO)"
            >
              下线
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="modelStore.total > 0"
        v-model:limit="modelStore.query.pageSize"
        v-model:page="modelStore.query.pageNum"
        v-model:total="modelStore.total"
        @pagination="modelStore.fetchModels()"
      />
    </el-card>

    <!-- 模型表单弹窗（类型化字段） -->
    <el-dialog
      v-model="modelStore.formDialog.visible"
      :title="formDialog.model ? '编辑模型' : '新增模型'"
      width="640px"
      destroy-on-close
      @closed="resetForm"
    >
      <el-form ref="formRef" :model="form" :rules="rules" label-width="130px">
        <el-form-item label="供应商" prop="providerId">
          <el-select
            v-model="form.providerId"
            class="!w-[240px]"
            placeholder="仅启用供应商"
          >
            <el-option
              v-for="p in enabledProviderOptions"
              :key="p.id"
              :label="p.displayName"
              :value="p.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="模型类型" prop="modelType">
          <el-select
            v-model="form.modelType"
            :disabled="!!formDialog.model"
            class="!w-[240px]"
          >
            <el-option label="对话 (chat)" value="chat" />
            <el-option label="向量 (embedding)" value="embedding" />
            <el-option label="重排 (rerank)" value="rerank" />
          </el-select>
        </el-form-item>
        <el-form-item label="模型标识" prop="modelId">
          <el-input
            v-model="form.modelId"
            :disabled="!!formDialog.model"
            placeholder="创建后不可修改，如 deepseek-chat"
          />
        </el-form-item>
        <el-form-item
          v-if="form.modelType === 'embedding'"
          label="向量维度"
          prop="dimension"
        >
          <el-input-number
            v-model="form.dimension"
            :min="1"
            :disabled="!!formDialog.model"
            controls-position="right"
          />
          <el-alert
            v-if="formDialog.model"
            class="ml-2"
            type="warning"
            :closable="false"
            title="维度创建后不可修改，变更需重建索引"
          />
        </el-form-item>
        <el-form-item label="显示名称" prop="displayName">
          <el-input v-model="form.displayName" />
        </el-form-item>
        <el-form-item label="最大上下文Token">
          <el-input-number
            v-model="form.maxContextTokens"
            :min="1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="最大输出Token">
          <el-input-number
            v-model="form.maxOutputTokens"
            :min="1"
            controls-position="right"
          />
        </el-form-item>

        <!-- chat 能力标识 -->
        <template v-if="form.modelType === 'chat'">
          <el-form-item label="能力标识">
            <el-checkbox v-model="form.supportsMultimodal">多模态</el-checkbox>
            <el-checkbox v-model="form.supportsToolCall">工具调用</el-checkbox>
            <el-checkbox v-model="form.supportsStreaming">流式输出</el-checkbox>
            <el-checkbox v-model="form.supportsPromptCache"
              >Prompt缓存</el-checkbox
            >
            <el-checkbox v-model="form.supportsStructuredOutput"
              >结构化输出</el-checkbox
            >
          </el-form-item>
          <el-form-item v-if="form.supportsPromptCache" label="缓存前缀长度">
            <el-input-number
              v-model="form.promptCachePrefixLen"
              :min="0"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="降级模型">
            <el-select
              v-model="form.fallbackModelId"
              class="!w-[240px]"
              clearable
              placeholder="可选，失败时降级到此模型"
            >
              <el-option
                v-for="m in modelStore.fallbackOptions"
                :key="m.id"
                :label="`${m.displayName} (${m.modelId})`"
                :value="m.id"
              />
            </el-select>
          </el-form-item>
        </template>

        <el-form-item label="VIP等级">
          <el-select v-model="form.vipLevel" class="!w-[240px]">
            <el-option label="所有用户" :value="0" />
            <el-option label="VIP1及以上" :value="1" />
            <el-option label="VIP2及以上" :value="2" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态">
          <el-switch
            v-model="form.status"
            :active-value="1"
            :inactive-value="0"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button type="primary" :loading="submitting" @click="submit"
          >确 定</el-button
        >
        <el-button @click="modelStore.formDialog.visible = false"
          >取 消</el-button
        >
      </template>
    </el-dialog>

    <!-- 价格档位矩阵 + 版本历史 -->
    <el-dialog
      v-model="modelStore.priceDialog.visible"
      :title="`价格档位 - ${priceDialog.model?.displayName ?? ''}`"
      width="820px"
      destroy-on-close
    >
      <div class="mb-2 text-xs text-gray-400">
        保存将生成新价格版本，历史版本自动保留；单价单位：积分/百万token
      </div>
      <el-table :data="modelStore.priceRows" size="small">
        <el-table-column label="Token类型" width="120">
          <template #default="{ row }">
            <el-select v-model="row.tokenType" size="small">
              <el-option label="input 输入" value="input" />
              <el-option label="cached 缓存命中" value="cached" />
              <el-option label="output 输出" value="output" />
            </el-select>
          </template>
        </el-table-column>
        <el-table-column label="时段" width="110">
          <template #default="{ row }">
            <el-select v-model="row.timeSlot" size="small">
              <el-option label="peak 高峰" value="peak" />
              <el-option label="idle 闲时" value="idle" />
            </el-select>
          </template>
        </el-table-column>
        <el-table-column label="分段下界" width="150">
          <template #default="{ row }">
            <el-input-number
              v-model="row.minTokens"
              :min="0"
              size="small"
              controls-position="right"
            />
          </template>
        </el-table-column>
        <el-table-column label="分段上界" width="150">
          <template #default="{ row }">
            <el-input-number
              v-model="row.maxTokens"
              :min="0"
              size="small"
              controls-position="right"
              placeholder="空为不限"
            />
          </template>
        </el-table-column>
        <el-table-column label="单价" width="150">
          <template #default="{ row }">
            <el-input-number
              v-model="row.unitPrice"
              :min="0"
              :precision="2"
              size="small"
              controls-position="right"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="70" align="center">
          <template #default="{ $index }">
            <el-button
              link
              type="danger"
              size="small"
              @click="modelStore.priceRows.splice($index, 1)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-button
        class="mt-2"
        size="small"
        type="primary"
        plain
        @click="addPriceRow"
      >
        <el-icon><Plus /></el-icon>添加档位
      </el-button>

      <el-divider content-position="left">版本历史</el-divider>
      <el-table
        v-loading="modelStore.priceLoading"
        :data="modelStore.priceHistory"
        size="small"
      >
        <el-table-column
          label="版本"
          prop="priceVersion"
          width="70"
          align="center"
        />
        <el-table-column label="生效时间" prop="effectiveFrom" width="160" />
        <el-table-column label="失效时间" width="160">
          <template #default="{ row }">{{
            row.effectiveTo ?? "长期"
          }}</template>
        </el-table-column>
        <el-table-column label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.status === 1 ? 'success' : 'info'" size="small">
              {{ row.status === 1 ? "生效" : "停用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="档位明细">
          <template #default="{ row }">
            <div v-for="d in row.details" :key="d.id" class="text-xs leading-5">
              {{ d.tokenType }}/{{ d.timeSlot }} [{{ d.minTokens }}~{{
                d.maxTokens ?? "∞"
              }}]： {{ d.unitPrice }} 积分/百万token
            </div>
          </template>
        </el-table-column>
      </el-table>
      <pagination
        v-if="modelStore.priceTotal > modelStore.priceQuery.size!"
        v-model:limit="modelStore.priceQuery.size"
        v-model:page="modelStore.priceQuery.page"
        v-model:total="modelStore.priceTotal"
        layout="prev, pager, next"
        @pagination="
          modelStore.fetchPriceHistory(modelStore.priceDialog.model!.modelId)
        "
      />

      <template #footer>
        <el-button
          type="primary"
          :loading="modelStore.priceSubmitting"
          @click="savePrice"
        >
          保存为新版本
        </el-button>
        <el-button @click="modelStore.priceDialog.visible = false"
          >取 消</el-button
        >
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "AiModelsModels" });

import { Plus, Refresh } from "@element-plus/icons-vue";
import { useDebounceFn } from "@vueuse/core";
import {
  AiModelForm,
  AiModelType,
  AiModelVO,
  ProviderHealth,
  ProviderVO,
} from "dehaze-sdk-js";
import { useAdminModelStore } from "@/store/modules/adminModel";
import { useAdminProviderStore } from "@/store/modules/adminProvider";

const modelStore = useAdminModelStore();
const providerStore = useAdminProviderStore();

const formRef = ref(ElForm);
const submitting = ref(false);

const formDialog = computed(() => modelStore.formDialog);
const priceDialog = computed(() => modelStore.priceDialog);
const operation = computed(() => modelStore.operation);

const typeTag: Record<
  AiModelType,
  { label: string; type: "primary" | "success" | "warning" }
> = {
  chat: { label: "对话", type: "primary" },
  embedding: { label: "向量", type: "success" },
  rerank: { label: "重排", type: "warning" },
};
const vipLabel = ["所有用户", "VIP1+", "VIP2+"];
const speedLabel: Record<string, string> = {
  fast: "快",
  medium: "中",
  slow: "慢",
};

function providerHealthTag(health: string) {
  return (
    providerStore.healthTagMap[health as ProviderHealth] ?? {
      label: health ?? "-",
      type: "info" as const,
    }
  );
}

const enabledProviderOptions = computed(() =>
  modelStore.providers.filter((p) => p.status === 1)
);

function handleQuery() {
  modelStore.query.pageNum = 1;
  modelStore.fetchModels();
}
const debouncedQuery = useDebounceFn(handleQuery, 300);

// ==================== 模型表单 ====================

const emptyForm = () => ({
  providerId: undefined as number | undefined,
  modelId: "",
  modelType: "chat" as AiModelType,
  dimension: undefined as number | undefined,
  displayName: "",
  maxContextTokens: 4096,
  maxOutputTokens: 4096,
  supportsMultimodal: false,
  supportsToolCall: false,
  supportsStreaming: true,
  supportsPromptCache: false,
  supportsStructuredOutput: false,
  promptCachePrefixLen: 0,
  fallbackModelId: null as number | null,
  vipLevel: 0,
  status: 1 as 0 | 1,
});
const form = reactive(emptyForm());

const rules = {
  providerId: [
    { required: true, message: "供应商不能为空", trigger: "change" },
  ],
  modelId: [{ required: true, message: "模型标识不能为空", trigger: "blur" }],
  displayName: [
    { required: true, message: "显示名称不能为空", trigger: "blur" },
  ],
};

watch(formDialog, ({ visible, model }) => {
  if (!visible) return;
  Object.assign(form, emptyForm());
  if (model) {
    Object.assign(form, {
      providerId: model.providerId,
      modelId: model.modelId,
      modelType: model.modelType,
      dimension: model.dimension ?? undefined,
      displayName: model.displayName,
      maxContextTokens: model.maxContextTokens,
      maxOutputTokens: model.maxOutputTokens,
      supportsMultimodal: model.supportsMultimodal === 1,
      supportsToolCall: model.supportsToolCall === 1,
      supportsStreaming: model.supportsStreaming === 1,
      supportsPromptCache: model.supportsPromptCache === 1,
      supportsStructuredOutput: model.supportsStructuredOutput === 1,
      promptCachePrefixLen: model.promptCachePrefixLen,
      fallbackModelId: model.fallbackModelId ?? null,
      vipLevel: model.vipLevel,
      status: model.status,
    });
  }
});

function resetForm() {
  Object.assign(form, emptyForm());
}

function openCreate() {
  modelStore.openFormDialog(null);
}
function openEdit(row: AiModelVO) {
  modelStore.openFormDialog(row);
}

async function submit() {
  await formRef.value.validate();

  // embedding 模型的维度决定向量索引映射，创建时必须明确
  if (form.modelType === "embedding" && !form.dimension) {
    ElMessage.error("embedding 模型必须指定向量维度");
    return;
  }

  submitting.value = true;
  try {
    const payload: AiModelForm = {
      providerId: form.providerId!,
      modelId: form.modelId,
      modelType: form.modelType,
      dimension: form.modelType === "embedding" ? form.dimension : undefined,
      displayName: form.displayName,
      maxContextTokens: form.maxContextTokens,
      maxOutputTokens: form.maxOutputTokens,
      supportsMultimodal: form.supportsMultimodal,
      supportsToolCall: form.supportsToolCall,
      supportsStreaming: form.supportsStreaming,
      supportsPromptCache: form.supportsPromptCache,
      supportsStructuredOutput: form.supportsStructuredOutput,
      promptCachePrefixLen: form.promptCachePrefixLen,
      fallbackModelId:
        form.modelType === "chat" ? form.fallbackModelId : undefined,
      vipLevel: form.vipLevel,
      status: form.status,
    };
    await modelStore.saveModel(payload);
    ElMessage.success("保存成功");
    modelStore.formDialog.visible = false;
  } finally {
    submitting.value = false;
  }
}

async function handleStatusChange(row: AiModelVO) {
  try {
    await ElMessageBox.confirm(
      `确认${row.status === 1 ? "启用" : "禁用"}模型「${row.displayName}」？禁用后消费方立即不可选。`,
      "状态变更",
      { type: "warning" }
    );
    await modelStore.toggleStatus(row, row.status as 0 | 1);
    ElMessage.success(row.status === 1 ? "已启用" : "已禁用");
  } catch (e) {
    // 取消确认或请求失败都回滚开关
    row.status = row.status === 1 ? 0 : 1;
    if (e instanceof Error) throw e;
  }
}

async function handleDelete(row: AiModelVO) {
  await ElMessageBox.confirm(
    `确认下线模型「${row.displayName}」？model_id 不可复用，历史会话引用将失效。`,
    "下线确认",
    { type: "warning" }
  );
  await modelStore.deleteModel(row);
  ElMessage.success("已下线");
}

// ==================== 价格档位 ====================

function addPriceRow() {
  modelStore.priceRows.push({
    tokenType: "input",
    timeSlot: "peak",
    minTokens: 0,
    maxTokens: null,
    unitPrice: 0,
  });
}

async function savePrice() {
  if (modelStore.priceRows.length === 0) {
    ElMessage.warning("请至少配置一条价格档位");
    return;
  }
  const invalidRow = modelStore.priceRows.find(
    (row) => row.unitPrice == null || row.minTokens == null
  );
  if (invalidRow) {
    ElMessage.error("价格档位存在未填写的单价或分段下界");
    return;
  }
  await modelStore.savePrice();
}

onMounted(() => {
  modelStore.fetchModels();
  modelStore.fetchProviders();
  modelStore.fetchOperation();
});
</script>
