<!-- 外部服务凭据配置：仅录入/更新，不回显明文（加密存储，不暴露给 LLM） -->
<template>
  <div>
    <el-alert
      class="mb-3"
      type="info"
      :closable="false"
      title="凭据加密存储，保存后不可回显、不可查看明文，仅可重新录入覆盖。"
    />
    <div class="mb-3">
      <span class="text-sm">当前状态：</span>
      <el-tag v-if="configured" type="success" size="small">已配置</el-tag>
      <el-tag v-else type="info" size="small">未配置</el-tag>
    </div>

    <el-form label-width="110px">
      <el-form-item label="API Key">
        <el-input
          v-model="form.apiKey"
          type="password"
          show-password
          autocomplete="new-password"
          placeholder="留空表示不更新该字段"
        />
      </el-form-item>
      <el-form-item label="其他凭据字段">
        <div class="w-full">
          <div
            v-for="(item, index) in extraRows"
            :key="index"
            class="mb-2 flex gap-2"
          >
            <el-input
              v-model="item.key"
              class="w-[140px]"
              placeholder="字段名"
            />
            <el-input
              v-model="item.value"
              type="password"
              show-password
              placeholder="字段值"
            />
            <el-button
              link
              type="danger"
              size="small"
              @click="extraRows.splice(index, 1)"
            >
              删除
            </el-button>
          </div>
          <el-button size="small" type="primary" plain @click="addExtraRow">
            <el-icon><Plus /></el-icon>添加字段
          </el-button>
        </div>
      </el-form-item>
    </el-form>

    <div class="flex justify-end">
      <el-button
        v-hasPerm="['ai:mcp:manage']"
        type="primary"
        :loading="submitting"
        @click="handleSubmit"
      >
        保存凭据
      </el-button>
    </div>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "CredentialPanel" });

import { Plus } from "@element-plus/icons-vue";
import { McpCredentialForm } from "dehaze-sdk-js";
import { useAdminMcpStore } from "@/store/modules/adminMcp";

const props = defineProps<{ serverId: number; configured: boolean }>();

const mcpStore = useAdminMcpStore();

const form = reactive<McpCredentialForm>({ apiKey: "" });
const extraRows = ref<{ key: string; value: string }[]>([]);
const submitting = ref(false);

watch(
  () => props.serverId,
  () => {
    form.apiKey = "";
    extraRows.value = [];
  }
);

function addExtraRow() {
  extraRows.value.push({ key: "", value: "" });
}

async function handleSubmit() {
  const extra: Record<string, string> = {};
  for (const row of extraRows.value) {
    if (!row.key.trim()) {
      ElMessage.warning("凭据字段名不能为空");
      return;
    }
    extra[row.key.trim()] = row.value;
  }
  if (!form.apiKey && Object.keys(extra).length === 0) {
    ElMessage.warning("请至少填写 API Key 或一条其他凭据字段");
    return;
  }
  submitting.value = true;
  try {
    await mcpStore.configureCredentials(props.serverId, {
      apiKey: form.apiKey || undefined,
      extra: Object.keys(extra).length > 0 ? extra : undefined,
    });
    form.apiKey = "";
    extraRows.value = [];
  } finally {
    submitting.value = false;
  }
}
</script>
