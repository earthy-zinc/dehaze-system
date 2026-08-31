<!-- 管理端智能体管理页：类型筛选 + 列表 + 新建/编辑/复制/测试弹窗 -->
<template>
  <div class="app-container">
    <el-card shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-2">
            <el-input
              v-model="agentStore.query.keyword"
              clearable
              placeholder="名称/编码"
              style="width: 180px"
              @keyup.enter="handleQuery"
              @input="debouncedQuery"
            />
            <AgentTypeFilter />
          </div>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            type="success"
            @click="
              agentStore.agentForm.agentId = null;
              agentStore.agentForm.visible = true;
            "
          >
            <el-icon><Plus /></el-icon>新建智能体
          </el-button>
        </div>
      </template>

      <AgentTable @edit="openEdit" @copy="openCopy" @test="openTest" />
    </el-card>

    <!-- 配置表单弹窗：编辑/新建共用 AgentConfigForm，保存生成草稿快照 -->
    <el-dialog
      v-model="agentStore.agentForm.visible"
      :title="agentStore.agentForm.agentId ? '编辑智能体' : '新建智能体'"
      width="860px"
      top="4vh"
      destroy-on-close
    >
      <AgentConfigForm
        :agent-id="agentStore.agentForm.agentId"
        @saved="agentStore.agentForm.visible = false"
      />
    </el-dialog>

    <!-- 复制弹窗：仅复制基本信息和配置，不复制关联关系；编码需重新指定 -->
    <el-dialog
      v-model="copyVisible"
      title="复制智能体"
      width="440px"
      destroy-on-close
    >
      <el-form label-width="100px">
        <el-form-item label="来源">
          <span>{{ copySource?.name }}（{{ copySource?.agentCode }}）</span>
        </el-form-item>
        <el-form-item label="新编码" required>
          <el-input
            v-model="copyAgentCode"
            placeholder="唯一编码，不可与现有 Agent 重复"
          />
        </el-form-item>
      </el-form>
      <el-alert
        type="info"
        :closable="false"
        title="复制基本信息和配置，不复制 Skills/MCP/子 Agent 关联关系。"
      />
      <template #footer>
        <el-button type="primary" :loading="copySubmitting" @click="handleCopy"
          >确 定</el-button
        >
        <el-button @click="copyVisible = false">取 消</el-button>
      </template>
    </el-dialog>

    <!-- 测试弹窗：即时预览响应，不入库不推送 -->
    <el-dialog
      v-model="testVisible"
      title="测试智能体"
      width="640px"
      destroy-on-close
    >
      <div class="mb-2 text-sm text-gray-500">
        {{ testSource?.name }}（{{ testSource?.agentCode }}）
      </div>
      <el-input
        v-model="testMessage"
        type="textarea"
        :rows="4"
        placeholder="输入测试消息"
      />
      <div class="mt-3">
        <el-button
          v-hasPerm="['ai:agent:manage']"
          type="primary"
          :loading="testSubmitting"
          :disabled="!testMessage"
          @click="handleTest"
        >
          运行测试
        </el-button>
        <span class="ml-2 text-xs text-gray-400"
          >即时预览不入库，消耗平台配额</span
        >
      </div>
      <pre v-if="testResult" class="test-result-pre">{{
        JSON.stringify(testResult, null, 2)
      }}</pre>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "AiAgents" });

import { Plus } from "@element-plus/icons-vue";
import { useDebounceFn } from "@vueuse/core";
import { AgentListItem } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";

const agentStore = useAdminAgentStore();

function handleQuery() {
  agentStore.query.pageNum = 1;
  agentStore.fetchAgents();
}
const debouncedQuery = useDebounceFn(handleQuery, 300);

onMounted(() => {
  agentStore.fetchAgents();
});

// ==================== 编辑 ====================
function openEdit(row: AgentListItem) {
  agentStore.agentForm.agentId = row.id;
  agentStore.agentForm.visible = true;
}

// ==================== 复制 ====================
const copyVisible = ref(false);
const copySource = ref<AgentListItem | null>(null);
const copyAgentCode = ref("");
const copySubmitting = ref(false);

function openCopy(row: AgentListItem) {
  copySource.value = row;
  // 预填编码避免重复，可修改；校验唯一性由后端 A0501 兜底
  copyAgentCode.value = `${row.agentCode}-copy`;
  copyVisible.value = true;
}

async function handleCopy() {
  if (!copyAgentCode.value) {
    ElMessage.error("新编码不能为空");
    return;
  }
  copySubmitting.value = true;
  try {
    await agentStore.copyAgent(copySource.value!.id, copyAgentCode.value);
    ElMessage.success("复制成功");
    copyVisible.value = false;
  } finally {
    copySubmitting.value = false;
  }
}

// ==================== 测试 ====================
const testVisible = ref(false);
const testSource = ref<AgentListItem | null>(null);
const testMessage = ref("");
const testSubmitting = ref(false);
const testResult = ref<Record<string, unknown> | null>(null);

function openTest(row: AgentListItem) {
  testSource.value = row;
  testMessage.value = "";
  testResult.value = null;
  testVisible.value = true;
}

async function handleTest() {
  testSubmitting.value = true;
  try {
    testResult.value = await agentStore.testAgent(
      testSource.value!.id,
      testMessage.value
    );
  } finally {
    testSubmitting.value = false;
  }
}
</script>

<style lang="scss" scoped>
.test-result-pre {
  max-height: 320px;
  padding: 8px 12px;
  margin: 12px 0 0;
  overflow: auto;
  font-size: 12px;
  line-height: 1.6;
  word-break: break-all;
  white-space: pre-wrap;
  background-color: var(--el-fill-color-light);
  border-radius: 4px;
}
</style>
