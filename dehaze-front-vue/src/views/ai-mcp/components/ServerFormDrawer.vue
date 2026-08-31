<!-- MCP Server 注册/编辑抽屉：协议配置 + 工具与命名空间 + 凭据 -->
<template>
  <el-drawer
    v-model="mcpStore.serverForm.visible"
    :title="server ? `Server 配置 · ${server.name}` : '注册 MCP Server'"
    size="720px"
    destroy-on-close
    @closed="resetForm"
  >
    <el-tabs
      :model-value="mcpStore.serverForm.tab"
      @update:model-value="handleTabChange"
    >
      <el-tab-pane label="配置" name="config">
        <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
          <el-form-item label="Server 名称" prop="name">
            <el-input
              v-model="form.name"
              placeholder="平台内唯一，如 github-mcp"
            />
          </el-form-item>
          <el-form-item label="描述">
            <el-input
              v-model="form.description"
              type="textarea"
              :rows="2"
              placeholder="用途与能力说明"
            />
          </el-form-item>
          <el-form-item label="传输协议" prop="protocolType">
            <el-select v-model="form.protocolType" class="w-full">
              <el-option
                v-for="(label, value) in MCP_PROTOCOL_LABELS"
                :key="value"
                :label="label"
                :value="value"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="端点 URL" prop="endpoint">
            <el-input
              v-model="form.endpoint"
              :placeholder="
                form.protocolType === 'stdio'
                  ? 'stdio 为本地进程，端点可留空'
                  : '如 https://mcp.example.com/mcp'
              "
            />
          </el-form-item>
          <el-form-item label="鉴权方式">
            <el-select v-model="form.authType" class="w-full">
              <el-option
                v-for="(label, value) in MCP_AUTH_LABELS"
                :key="value"
                :label="label"
                :value="value"
              />
            </el-select>
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="工具与命名空间" name="tools" :disabled="!server">
        <ToolNamespacePanel v-if="server" :server-id="server.id" />
      </el-tab-pane>

      <el-tab-pane label="凭据" name="credentials" :disabled="!server">
        <CredentialPanel
          v-if="server"
          :server-id="server.id"
          :configured="server.credentialConfigured ?? false"
        />
      </el-tab-pane>
    </el-tabs>

    <template #footer>
      <div class="flex justify-between items-center">
        <div v-if="server" class="flex items-center gap-2">
          <span class="text-sm">启用状态</span>
          <el-switch
            v-hasPerm="['ai:mcp:manage']"
            :model-value="server.status"
            :active-value="1"
            :inactive-value="0"
            @change="handleStatusChange"
          />
          <span class="text-xs text-gray-400">预览工具清单后再启用</span>
        </div>
        <div>
          <el-button @click="mcpStore.serverForm.visible = false"
            >关闭</el-button
          >
          <el-button
            v-if="mcpStore.serverForm.tab === 'config'"
            v-hasPerm="['ai:mcp:manage']"
            type="primary"
            :loading="submitting"
            @click="handleSubmit"
          >
            {{ server ? "保存配置" : "注册并发现工具" }}
          </el-button>
        </div>
      </div>
    </template>
  </el-drawer>
</template>

<script lang="ts" setup>
defineOptions({ name: "ServerFormDrawer" });

import type { FormInstance, FormRules } from "element-plus";
import { McpServerForm } from "dehaze-sdk-js";
import CredentialPanel from "./CredentialPanel.vue";
import ToolNamespacePanel from "./ToolNamespacePanel.vue";
import {
  MCP_AUTH_LABELS,
  MCP_PROTOCOL_LABELS,
  useAdminMcpStore,
} from "@/store/modules/adminMcp";

const mcpStore = useAdminMcpStore();

const formRef = ref<FormInstance>();
const submitting = ref(false);

const server = computed(() => mcpStore.serverForm.server);

function emptyForm(): McpServerForm {
  return {
    name: "",
    description: "",
    protocolType: "streamable-http",
    endpoint: "",
    authType: "none",
  };
}

const form = reactive<McpServerForm>(emptyForm());

const rules: FormRules<McpServerForm> = {
  name: [{ required: true, message: "Server 名称不能为空", trigger: "blur" }],
  protocolType: [
    { required: true, message: "传输协议不能为空", trigger: "change" },
  ],
  endpoint: [
    {
      validator: (_rule, value: string, callback) => {
        if (form.protocolType === "stdio") {
          callback();
          return;
        }
        if (!value) {
          callback(new Error("端点 URL 不能为空"));
          return;
        }
        let url: URL;
        try {
          url = new URL(value);
        } catch {
          callback(new Error("端点 URL 格式不合法"));
          return;
        }
        if (url.protocol !== "http:" && url.protocol !== "https:") {
          callback(new Error("端点仅支持 http/https"));
          return;
        }
        callback();
      },
      trigger: "blur",
    },
  ],
};

watch(
  () => [mcpStore.serverForm.visible, mcpStore.serverForm.server] as const,
  ([visible, current]) => {
    if (!visible) return;
    Object.assign(form, emptyForm());
    if (current) {
      Object.assign(form, {
        name: current.name,
        description: current.description ?? "",
        protocolType: current.protocolType,
        endpoint: current.endpoint ?? "",
        authType: current.authType ?? "none",
      });
    }
  },
  { immediate: true }
);

function resetForm() {
  Object.assign(form, emptyForm());
}

function handleTabChange(tab: string | number) {
  mcpStore.switchDrawerTab(tab as "config" | "tools" | "credentials");
}

async function handleSubmit() {
  await formRef.value?.validate();
  submitting.value = true;
  try {
    await mcpStore.registerServer({ ...form });
  } finally {
    submitting.value = false;
  }
}

async function handleStatusChange(status: string | number | boolean) {
  const current = server.value;
  if (!current) return;
  await mcpStore.switchServerStatus(current, status === 1 ? 1 : 0);
  ElMessage.success(status === 1 ? "已启用" : "已禁用");
}
</script>
