<!-- 召回测试：测试集管理（问题+期望命中段落）+ 执行结果 Recall@K/命中率对比 -->
<script lang="ts" setup>
import { RecallTestResultVO } from "dehaze-sdk-js";
import { onMounted, reactive, ref } from "vue";
import { Plus } from "@element-plus/icons-vue";
import { useAdminKbStore } from "@/store/modules/adminKb";

defineOptions({ name: "RecallTestPanel" });

const props = defineProps<{
  kbId: number;
}>();

const adminKbStore = useAdminKbStore();

const createDialog = reactive({
  visible: false,
  question: "",
  // 期望命中分块 ID，逗号分隔输入
  expectedChunkIds: "",
  submitting: false,
});
const runningId = ref<number | null>(null);

// 本会话内历次执行结果，用于调参后重跑对比
const runHistory = ref<RecallTestResultVO[]>([]);

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function openCreateDialog() {
  createDialog.question = "";
  createDialog.expectedChunkIds = "";
  createDialog.visible = true;
}

async function submitCreate() {
  if (!createDialog.question.trim()) {
    ElMessage.warning("请输入测试问题");
    return;
  }
  const expectedChunkIds = createDialog.expectedChunkIds
    .split(/[,，\s]+/)
    .map(Number)
    .filter((n) => Number.isInteger(n) && n > 0);
  if (expectedChunkIds.length === 0) {
    ElMessage.warning("请输入期望命中的分块 ID（逗号分隔）");
    return;
  }
  createDialog.submitting = true;
  try {
    await adminKbStore.createTestSet(props.kbId, {
      question: createDialog.question.trim(),
      expectedChunkIds,
    });
    ElMessage.success("测试集已创建");
    createDialog.visible = false;
  } finally {
    createDialog.submitting = false;
  }
}

async function handleRun(testSetId: number) {
  runningId.value = testSetId;
  try {
    const result = await adminKbStore.runRecallSet(props.kbId, testSetId);
    if (result) {
      runHistory.value.unshift(result);
    }
  } finally {
    runningId.value = null;
  }
}

onMounted(() => {
  adminKbStore.fetchRecallSets(props.kbId);
});
</script>

<template>
  <div>
    <div class="flex justify-end mb-3">
      <el-button
        v-has-perm="['kb:audit']"
        type="primary"
        @click="openCreateDialog"
      >
        <el-icon><Plus /></el-icon>
        新建测试集
      </el-button>
    </div>

    <el-table :data="adminKbStore.recallSets" border size="small">
      <el-table-column
        label="测试问题"
        prop="question"
        min-width="200"
        show-overflow-tooltip
      />
      <el-table-column label="期望命中分块" min-width="160">
        <template #default="{ row }">
          <el-tag
            v-for="chunkId in (row as { expectedChunkIds: number[] })
              .expectedChunkIds"
            :key="chunkId"
            size="small"
            class="mr-1"
          >
            #{{ chunkId }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="创建时间"
        width="170"
        align="center"
        prop="createTime"
      />
      <el-table-column label="操作" width="120" align="center">
        <template #default="{ row }">
          <el-button
            v-has-perm="['kb:audit']"
            size="small"
            link
            type="primary"
            :loading="runningId === (row as { id: number }).id"
            @click="handleRun((row as { id: number }).id)"
          >
            执行
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <template v-if="runHistory.length > 0">
      <el-divider content-position="left"
        >执行结果（调参后重跑可对比）</el-divider
      >
      <el-table :data="runHistory" border size="small">
        <el-table-column
          label="测试集 ID"
          prop="testSetId"
          width="110"
          align="center"
        />
        <el-table-column label="Recall@K" width="120" align="center">
          <template #default="{ row }">
            {{ percent((row as RecallTestResultVO).recallAtK) }}
          </template>
        </el-table-column>
        <el-table-column label="命中率" width="120" align="center">
          <template #default="{ row }">
            {{ percent((row as RecallTestResultVO).hitRate) }}
          </template>
        </el-table-column>
        <el-table-column
          label="总用例"
          prop="totalCases"
          width="100"
          align="center"
        />
        <el-table-column
          label="命中用例"
          prop="hitCases"
          width="100"
          align="center"
        />
      </el-table>
    </template>

    <el-dialog
      v-model="createDialog.visible"
      title="新建召回测试集"
      width="520px"
      append-to-body
    >
      <el-form label-width="110px">
        <el-form-item label="测试问题" required>
          <el-input
            v-model="createDialog.question"
            placeholder="输入用于召回测试的问题"
          />
        </el-form-item>
        <el-form-item label="期望命中分块" required>
          <el-input
            v-model="createDialog.expectedChunkIds"
            placeholder="期望命中的分块 ID，逗号分隔，如 101,102"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialog.visible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="createDialog.submitting"
          @click="submitCreate"
        >
          创建
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>
