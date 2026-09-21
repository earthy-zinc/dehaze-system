<!-- 评测集与样本管理：评测集 CRUD + 样本 CRUD（删除均二次确认） -->
<template>
  <el-card shadow="never" class="mb-[12px]">
    <template #header>
      <div class="flex justify-between items-center">
        <span>评测集与样本</span>
        <el-button
          v-hasPerm="['ai:agent:manage']"
          type="primary"
          plain
          size="small"
          @click="openDatasetDialog()"
        >
          <el-icon><Plus /></el-icon>新建评测集
        </el-button>
      </div>
    </template>

    <el-empty
      v-if="props.agentId == null"
      description="请先选择智能体管理评测集与样本"
      :image-size="60"
    />
    <el-table v-else :data="agentStore.evalDatasets" size="small">
      <el-table-column label="名称" prop="name" min-width="160" />
      <el-table-column label="类型" width="110" align="center">
        <template #default="{ row }">
          <el-tag :type="datasetMeta(row.datasetType).type" size="small">
            {{ datasetMeta(row.datasetType).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="描述"
        prop="description"
        min-width="200"
        show-overflow-tooltip
      />
      <el-table-column label="创建时间" prop="createTime" width="170" />
      <el-table-column label="操作" width="260" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click="openSamples(row as EvalDatasetResult)"
          >
            样本
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="openSampleDialog(row.id)"
          >
            新增样本
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="openDatasetDialog(row as EvalDatasetResult)"
          >
            编辑
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="danger"
            size="small"
            @click="handleDeleteDataset(row as EvalDatasetResult)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="暂无评测集，请先新建" :image-size="60" />
      </template>
    </el-table>

    <!-- 样本列表抽屉 -->
    <el-drawer
      v-model="samplesVisible"
      :title="`样本管理 · ${currentDataset?.name ?? ''}`"
      size="60%"
      destroy-on-close
    >
      <div class="mb-[12px]">
        <el-button
          v-hasPerm="['ai:agent:manage']"
          type="primary"
          plain
          size="small"
          @click="openSampleDialog(currentDataset?.id ?? null)"
        >
          <el-icon><Plus /></el-icon>新建样本
        </el-button>
        <span class="ml-2 text-xs text-gray-400">
          共 {{ agentStore.evalSamples.length }} 条样本
        </span>
      </div>

      <el-table
        v-loading="agentStore.evalSamplesLoading"
        :data="agentStore.evalSamples"
        size="small"
      >
        <el-table-column label="任务目标" prop="taskGoal" min-width="220" />
        <el-table-column label="风险" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="riskMeta(row.riskLevel).type" size="small">
              {{ riskMeta(row.riskLevel).label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="期望结果"
          prop="expectedResult"
          min-width="180"
          show-overflow-tooltip
        />
        <el-table-column label="操作" width="140" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              v-hasPerm="['ai:agent:manage']"
              link
              type="primary"
              size="small"
              @click="openSampleDialog(row.datasetId, row as EvalSampleResult)"
            >
              编辑
            </el-button>
            <el-button
              v-hasPerm="['ai:agent:manage']"
              link
              type="danger"
              size="small"
              @click="handleDeleteSample(row as EvalSampleResult)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="该评测集暂无样本" :image-size="60" />
        </template>
      </el-table>
    </el-drawer>

    <!-- 评测集表单 -->
    <el-dialog
      v-model="datasetVisible"
      :title="editingDataset ? '编辑评测集' : '新建评测集'"
      width="560px"
      destroy-on-close
    >
      <el-form
        ref="datasetFormRef"
        :model="datasetForm"
        :rules="datasetRules"
        label-width="100px"
      >
        <el-form-item label="名称" prop="name">
          <el-input v-model="datasetForm.name" maxlength="64" show-word-limit />
        </el-form-item>
        <el-form-item v-if="!editingDataset" label="类型" prop="datasetType">
          <el-select v-model="datasetForm.datasetType" class="!w-full">
            <el-option
              v-for="option in DATASET_TYPE_OPTIONS"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="描述">
          <el-input
            v-model="datasetForm.description"
            type="textarea"
            :rows="3"
            maxlength="200"
            show-word-limit
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button
          type="primary"
          :loading="datasetSubmitting"
          @click="handleDatasetSubmit"
        >
          确 定
        </el-button>
        <el-button @click="datasetVisible = false">取 消</el-button>
      </template>
    </el-dialog>

    <EvalSampleFormDialog
      v-if="props.agentId != null"
      v-model="sampleVisible"
      :agent-id="props.agentId"
      :dataset-id="sampleDatasetId"
      :sample="editingSample"
      @saved="handleSampleSaved"
    />
  </el-card>
</template>

<script lang="ts" setup>
import { Plus } from "@element-plus/icons-vue";
import type {
  EvalDatasetResult,
  EvalDatasetType,
  EvalSampleResult,
} from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";
import EvalSampleFormDialog from "./EvalSampleFormDialog.vue";
import {
  DATASET_TYPE_META,
  DATASET_TYPE_OPTIONS,
  RISK_LEVEL_META,
} from "../eval-meta";

defineOptions({ name: "EvalDatasetPanel" });

const props = defineProps<{ agentId: number | null }>();

const agentStore = useAdminAgentStore();

// ==================== 评测集 ====================
const datasetVisible = ref(false);
const datasetSubmitting = ref(false);
const datasetFormRef = ref(ElForm);
const editingDataset = ref<EvalDatasetResult | null>(null);
const datasetForm = reactive({
  name: "",
  description: "",
  datasetType: "regression" as EvalDatasetType,
});
const datasetRules = {
  name: [{ required: true, message: "评测集名称不能为空", trigger: "blur" }],
  datasetType: [
    { required: true, message: "请选择评测集类型", trigger: "change" },
  ],
};

function openDatasetDialog(dataset?: EvalDatasetResult) {
  editingDataset.value = dataset ?? null;
  datasetForm.name = dataset?.name ?? "";
  datasetForm.description = dataset?.description ?? "";
  datasetForm.datasetType = (dataset?.datasetType ??
    "regression") as EvalDatasetType;
  datasetVisible.value = true;
}

async function handleDatasetSubmit() {
  await datasetFormRef.value.validate();
  datasetSubmitting.value = true;
  try {
    if (editingDataset.value) {
      await agentStore.updateEvalDataset(
        props.agentId!,
        editingDataset.value.id,
        {
          name: datasetForm.name,
          description: datasetForm.description,
        }
      );
      ElMessage.success("评测集已更新");
    } else {
      await agentStore.createEvalDataset(props.agentId!, {
        name: datasetForm.name,
        description: datasetForm.description,
        datasetType: datasetForm.datasetType,
      });
      ElMessage.success("评测集已创建");
    }
    datasetVisible.value = false;
  } finally {
    datasetSubmitting.value = false;
  }
}

async function handleDeleteDataset(dataset: EvalDatasetResult) {
  await ElMessageBox.confirm(
    `确认删除评测集「${dataset.name}」？其下样本将一并删除，历史评测记录不受影响。`,
    "删除确认",
    { type: "warning" }
  );
  await agentStore.deleteEvalDataset(props.agentId!, dataset.id);
  ElMessage.success("评测集已删除");
}

// ==================== 样本 ====================
const samplesVisible = ref(false);
const sampleVisible = ref(false);
const currentDataset = ref<EvalDatasetResult | null>(null);
const editingSample = ref<EvalSampleResult | null>(null);
/** 新增样本的目标评测集，从样本详情行进入时为空（由表单选择） */
const sampleDatasetId = ref<number | null>(null);

async function openSamples(dataset: EvalDatasetResult) {
  currentDataset.value = dataset;
  samplesVisible.value = true;
  await agentStore.fetchEvalSamples(props.agentId!, dataset.id);
}

function openSampleDialog(datasetId: number | null, sample?: EvalSampleResult) {
  sampleDatasetId.value = datasetId;
  editingSample.value = sample ?? null;
  sampleVisible.value = true;
}

async function handleDeleteSample(sample: EvalSampleResult) {
  try {
    await ElMessageBox.confirm(
      `确认删除样本「${sample.taskGoal}」？已从该样本生成的评测记录不受影响。`,
      "删除确认",
      { type: "warning" }
    );
  } catch {
    return;
  }
  await agentStore.deleteEvalSample(
    props.agentId!,
    sample.datasetId,
    sample.id
  );
  ElMessage.success("样本已删除");
}

function handleSampleSaved() {
  if (currentDataset.value) {
    agentStore.fetchEvalSamples(props.agentId!, currentDataset.value.id);
  }
}

// 切换 Agent 时关闭抽屉/弹窗并丢弃上一个 Agent 的样本态
watch(
  () => props.agentId,
  (agentId) => {
    samplesVisible.value = false;
    sampleVisible.value = false;
    currentDataset.value = null;
    editingSample.value = null;
    if (agentId == null) return;
    agentStore.fetchEvalDatasets(agentId);
  },
  { immediate: true }
);

function datasetMeta(type: string) {
  return DATASET_TYPE_META[type] ?? { label: type, type: "info" as const };
}

function riskMeta(riskLevel: string) {
  return (
    RISK_LEVEL_META[riskLevel] ?? { label: riskLevel, type: "info" as const }
  );
}
</script>
