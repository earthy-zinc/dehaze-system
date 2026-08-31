<!-- A2A 端点管理区：Agent 对外暴露开关 + 外部端点注册/刷新/删除 -->
<template>
  <div>
    <el-card shadow="never" class="mb-3">
      <div class="flex items-center gap-3">
        <span class="font-medium">对外暴露</span>
        <el-switch
          v-model="exposed"
          :disabled="exposedSwitching"
          v-hasPerm="['ai:agent:manage']"
          @change="handleExposedChange"
        />
        <span class="text-xs text-gray-400">
          开启后平台提供 A2A 协议端点与 Agent Card，供外部 Agent 将本 Agent
          作为远程子 Agent 调用
        </span>
      </div>
    </el-card>

    <el-card shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <span>外部 A2A 端点</span>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            type="primary"
            size="small"
            @click="openCreate"
          >
            <el-icon><Plus /></el-icon>注册端点
          </el-button>
        </div>
      </template>
      <A2aEndpointTable @edit="openEdit" />
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="editingEndpoint ? '编辑外部 A2A 端点' : '注册外部 A2A 端点'"
      width="560px"
      destroy-on-close
    >
      <el-form ref="formRef" :model="form" :rules="rules" label-width="110px">
        <el-form-item label="端点名称" prop="name">
          <el-input v-model="form.name" />
        </el-form-item>
        <el-form-item label="端点地址" prop="baseUrl">
          <el-input
            v-model="form.baseUrl"
            placeholder="https://..."
            :disabled="!!editingEndpoint"
          />
        </el-form-item>
        <el-form-item label="Agent Card 地址">
          <el-input
            v-model="form.agentCardUrl"
            placeholder="可选，默认 {baseUrl}/.well-known/agent-card.json"
          />
        </el-form-item>
        <el-form-item label="认证方式">
          <el-select v-model="form.authType" class="!w-[240px]">
            <el-option label="API Key" value="apiKey" />
            <el-option label="HTTP 认证" value="http" />
            <el-option label="OAuth2" value="oauth2" />
            <el-option label="OpenID Connect" value="openIdConnect" />
            <el-option label="双向 TLS" value="mutualTLS" />
          </el-select>
        </el-form-item>
        <el-form-item label="凭证">
          <el-input
            v-model="form.credential"
            type="password"
            show-password
            placeholder="AES 加密存储，不回显"
          />
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
        <el-button @click="dialogVisible = false">取 消</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { Plus } from "@element-plus/icons-vue";
import {
  A2AAuthType,
  EndpointCreateForm,
  EndpointResult,
  EndpointUpdateForm,
} from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "A2aPanel" });

const props = defineProps<{ agentId: number }>();

const agentStore = useAdminAgentStore();

const exposed = ref(false);
const exposedSwitching = ref(false);
const dialogVisible = ref(false);
const submitting = ref(false);
const formRef = ref(ElForm);
const editingEndpoint = ref<EndpointResult | null>(null);

const emptyForm = () => ({
  name: "",
  baseUrl: "",
  agentCardUrl: "",
  authType: "apiKey" as A2AAuthType,
  credential: "",
  status: 1 as 0 | 1,
});
const form = reactive(emptyForm());

const rules = {
  name: [{ required: true, message: "端点名称不能为空", trigger: "blur" }],
  baseUrl: [{ required: true, message: "端点地址不能为空", trigger: "blur" }],
};

watch(
  () => props.agentId,
  async () => {
    const detail = await agentStore.fetchAgentDetail(props.agentId);
    exposed.value = detail.isExposed === 1;
  },
  { immediate: true }
);

async function handleExposedChange(value: boolean | string | number) {
  exposedSwitching.value = true;
  try {
    await agentStore.switchAgentExposed(props.agentId, Boolean(value));
    ElMessage.success(value ? "已开启对外暴露" : "已关闭对外暴露");
  } catch (e) {
    exposed.value = !exposed.value;
    if (e instanceof Error) throw e;
  } finally {
    exposedSwitching.value = false;
  }
}

function openCreate() {
  editingEndpoint.value = null;
  Object.assign(form, emptyForm());
  dialogVisible.value = true;
}

/** 编辑端点：地址不可改（后端不支持更新 baseUrl），凭证留空表示不更新 */
function openEdit(row: EndpointResult) {
  editingEndpoint.value = row;
  Object.assign(form, {
    name: row.name,
    baseUrl: row.baseUrl,
    agentCardUrl: row.agentCardUrl ?? "",
    authType: row.authType as A2AAuthType,
    credential: "",
    status: row.status,
  });
  dialogVisible.value = true;
}

async function submit() {
  await formRef.value.validate();
  submitting.value = true;
  try {
    if (editingEndpoint.value) {
      const payload: EndpointUpdateForm = {
        name: form.name,
        agentCardUrl: form.agentCardUrl || null,
        authType: form.authType,
        credential: form.credential || null,
        status: form.status,
      };
      await agentStore.manageA2aEndpoints("update", {
        id: editingEndpoint.value.id,
        form: payload,
      });
      ElMessage.success("端点已更新");
    } else {
      const payload: EndpointCreateForm = {
        name: form.name,
        baseUrl: form.baseUrl,
        agentCardUrl: form.agentCardUrl || null,
        authType: form.authType,
        credential: form.credential || null,
        status: form.status,
      };
      await agentStore.manageA2aEndpoints("create", { form: payload });
      ElMessage.success("端点已注册，Agent Card 已拉取");
    }
    dialogVisible.value = false;
  } finally {
    submitting.value = false;
  }
}
</script>
