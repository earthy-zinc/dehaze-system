<template>
  <el-drawer
    :model-value="modelValue"
    :title="
      isEdit
        ? `供应商配置 - ${form.displayName || provider?.displayName}`
        : '新增供应商'
    "
    size="640px"
    destroy-on-close
    @update:model-value="(v: boolean) => emit('update:modelValue', v)"
    @closed="resetForm"
  >
    <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
      <el-form-item label="编码" prop="providerCode">
        <el-input
          v-model="form.providerCode"
          :disabled="isEdit"
          placeholder="创建后不可修改"
        />
      </el-form-item>
      <el-form-item label="名称" prop="displayName">
        <el-input v-model="form.displayName" />
      </el-form-item>
      <el-form-item label="API地址" prop="apiBaseUrl">
        <el-input
          v-model="form.apiBaseUrl"
          placeholder="如 https://api.deepseek.com"
        />
      </el-form-item>
      <el-form-item label="协议类型" prop="protocolType">
        <el-select v-model="form.protocolType" class="!w-[200px]">
          <el-option label="openai_compat" value="openai_compat" />
          <el-option label="anthropic" value="anthropic" />
        </el-select>
      </el-form-item>
      <el-form-item label="认证方式" prop="authType">
        <el-select v-model="form.authType" class="!w-[200px]">
          <el-option label="bearer" value="bearer" />
          <el-option label="header" value="header" />
        </el-select>
      </el-form-item>
      <el-form-item label="默认请求头" prop="defaultHeaders">
        <el-input
          v-model="headersText"
          type="textarea"
          :rows="3"
          placeholder='JSON 格式，如 {"X-Custom": "v"}'
        />
      </el-form-item>
      <el-form-item label="健康检查">
        <el-switch
          v-model="form.healthCheckEnabled"
          :active-value="1"
          :inactive-value="0"
        />
        <span class="ml-2 text-xs text-gray-400">开启后参与熔断判定</span>
      </el-form-item>
      <el-form-item label="用户身份透传">
        <el-switch v-model="uifEnabled" />
      </el-form-item>
      <template v-if="uifEnabled">
        <el-form-item label="透传字段名" prop="uifField">
          <el-input
            v-model="form.uifField"
            placeholder="user_id / user / metadata.user_id"
          />
        </el-form-item>
        <el-form-item label="透传前缀">
          <el-input v-model="form.uifPrefix" placeholder="可选" />
        </el-form-item>
        <el-form-item label="最大长度">
          <el-input-number
            v-model="form.uifMaxLen"
            :min="1"
            :max="256"
            controls-position="right"
          />
        </el-form-item>
      </template>
      <el-form-item label="排序">
        <el-input-number
          v-model="form.sortOrder"
          :min="0"
          controls-position="right"
        />
      </el-form-item>
      <el-form-item label="状态">
        <el-switch
          v-model="form.status"
          :active-value="1"
          :inactive-value="0"
        />
      </el-form-item>
      <el-form-item label="备注">
        <el-input v-model="form.remark" type="textarea" :rows="2" />
      </el-form-item>
    </el-form>

    <el-alert
      v-if="isEdit && testResult"
      class="mb-4"
      type="success"
      :closable="false"
      title="最近一次连通性测试通过"
    />

    <!-- API Key 管理：仅编辑态可用（Key 挂在已存在供应商下） -->
    <template v-if="isEdit && provider">
      <el-divider content-position="left">API Key 管理</el-divider>

      <el-table
        v-loading="providerStore.keysLoading"
        :data="providerStore.keys"
        size="small"
      >
        <el-table-column label="名称" prop="name" min-width="90" />
        <el-table-column label="前缀" width="100">
          <template #default="{ row }">{{ row.keyPrefix || "-" }}</template>
        </el-table-column>
        <el-table-column label="状态" width="70" align="center">
          <template #default="{ row }">
            <el-switch
              v-model="row.status"
              :active-value="1"
              :inactive-value="0"
              size="small"
              @change="handleKeyStatusChange(row as ProviderKeyVO)"
            />
          </template>
        </el-table-column>
        <el-table-column
          label="优先级"
          prop="priority"
          width="70"
          align="center"
        />
        <el-table-column label="权重" prop="weight" width="60" align="center" />
        <el-table-column label="日限额" width="80" align="center">
          <template #default="{ row }">{{ row.dailyQuota ?? "∞" }}</template>
        </el-table-column>
        <el-table-column label="RPM" width="70" align="center">
          <template #default="{ row }">{{ row.rpmLimit ?? "-" }}</template>
        </el-table-column>
        <el-table-column label="过期时间" width="110">
          <template #default="{ row }">{{ row.expiresAt || "-" }}</template>
        </el-table-column>
        <el-table-column label="最近使用" width="150">
          <template #default="{ row }">{{ row.lastUsedAt || "-" }}</template>
        </el-table-column>
        <el-table-column label="操作" width="100" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              link
              type="primary"
              size="small"
              @click="openKeyEdit(row as ProviderKeyVO)"
              >编辑</el-button
            >
            <el-button
              link
              type="danger"
              size="small"
              @click="handleKeyDelete(row as ProviderKeyVO)"
              >删除</el-button
            >
          </template>
        </el-table-column>
      </el-table>

      <el-button
        class="mt-2"
        size="small"
        type="primary"
        plain
        @click="keyDialog.mode = 'create'"
      >
        <el-icon><Plus /></el-icon>新增 Key
      </el-button>

      <!-- Key 新增：明文仅提交不回显；编辑：仅非敏感字段 -->
      <el-dialog
        v-model="keyDialog.visible"
        :title="keyDialog.mode === 'create' ? '新增 API Key' : '编辑 API Key'"
        width="480px"
        append-to-body
      >
        <el-form
          ref="keyFormRef"
          :model="keyForm"
          :rules="keyRules"
          label-width="90px"
        >
          <el-form-item
            v-if="keyDialog.mode === 'create'"
            label="名称"
            prop="name"
          >
            <el-input v-model="keyForm.name" />
          </el-form-item>
          <el-form-item
            v-if="keyDialog.mode === 'create'"
            label="Key 明文"
            prop="key"
          >
            <el-input
              v-model="keyForm.key"
              type="password"
              show-password
              placeholder="仅提交存储，不回显"
            />
          </el-form-item>
          <el-form-item label="优先级" prop="priority">
            <el-input-number
              v-model="keyForm.priority"
              :min="0"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="权重" prop="weight">
            <el-input-number
              v-model="keyForm.weight"
              :min="0"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="日限额">
            <el-input-number
              v-model="keyForm.dailyQuota"
              :min="0"
              controls-position="right"
            />
            <span class="ml-2 text-xs text-gray-400">0 表示不限</span>
          </el-form-item>
          <el-form-item label="RPM">
            <el-input-number
              v-model="keyForm.rpmLimit"
              :min="0"
              controls-position="right"
            />
            <span class="ml-2 text-xs text-gray-400">0 表示不限</span>
          </el-form-item>
          <el-form-item label="过期时间">
            <el-date-picker
              v-model="keyForm.expiresAt"
              type="datetime"
              value-format="YYYY-MM-DD HH:mm:ss"
              placeholder="可选"
            />
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button type="primary" :loading="keySubmitting" @click="submitKey"
            >确 定</el-button
          >
          <el-button @click="keyDialog.visible = false">取 消</el-button>
        </template>
      </el-dialog>
    </template>

    <template #footer>
      <el-button type="primary" :loading="submitting" @click="submit"
        >保 存</el-button
      >
      <el-button @click="emit('update:modelValue', false)">取 消</el-button>
    </template>
  </el-drawer>
</template>

<script lang="ts" setup>
defineOptions({ name: "AiModelsProviderDrawer" });

import { Plus } from "@element-plus/icons-vue";
import { ProviderKeyVO, ProviderVO } from "dehaze-sdk-js";
import { useAdminProviderStore } from "@/store/modules/adminProvider";

const props = defineProps<{
  modelValue: boolean;
  provider: ProviderVO | null;
}>();
const emit = defineEmits<{ (e: "update:modelValue", v: boolean): void }>();

const providerStore = useAdminProviderStore();
const isEdit = computed(() => !!props.provider);

const formRef = ref(ElForm);
const keyFormRef = ref(ElForm);
const submitting = ref(false);
const keySubmitting = ref(false);

const emptyForm = () => ({
  providerCode: "",
  displayName: "",
  apiBaseUrl: "",
  protocolType: "openai_compat",
  authType: "bearer",
  sortOrder: 0,
  healthCheckEnabled: 1 as 0 | 1,
  status: 1 as 0 | 1,
  remark: "",
  uifField: "",
  uifPrefix: "",
  uifMaxLen: undefined as number | undefined,
});
const form = reactive(emptyForm());
const headersText = ref("{}");
const uifEnabled = ref(false);
const testResult = computed(() => providerStore.drawer.testResult);

const rules = {
  providerCode: [{ required: true, message: "编码不能为空", trigger: "blur" }],
  displayName: [{ required: true, message: "名称不能为空", trigger: "blur" }],
  apiBaseUrl: [{ required: true, message: "API地址不能为空", trigger: "blur" }],
};

watch(
  () => [props.modelValue, props.provider] as const,
  ([visible, provider]) => {
    if (!visible) return;
    Object.assign(form, emptyForm(), {
      ...(provider
        ? {
            providerCode: provider.providerCode,
            displayName: provider.displayName,
            apiBaseUrl: provider.apiBaseUrl,
            protocolType: provider.protocolType,
            authType: provider.authType,
            sortOrder: provider.sortOrder,
            healthCheckEnabled: provider.healthCheckEnabled,
            status: provider.status,
            remark: provider.remark ?? "",
          }
        : {}),
    });
    headersText.value = provider?.defaultHeaders
      ? JSON.stringify(provider.defaultHeaders, null, 2)
      : "{}";
    uifEnabled.value = !!provider?.userIdentityForward?.enabled;
    form.uifField = provider?.userIdentityForward?.field ?? "";
    form.uifPrefix = provider?.userIdentityForward?.prefix ?? "";
    form.uifMaxLen = provider?.userIdentityForward?.maxLen;
  },
  { immediate: true }
);

function resetForm() {
  Object.assign(form, emptyForm());
  headersText.value = "{}";
  uifEnabled.value = false;
}

async function submit() {
  await formRef.value.validate();

  // 请求头 JSON 在前端先校验，避免保存后才暴露格式错误
  let defaultHeaders: Record<string, unknown> | null = null;
  if (headersText.value.trim()) {
    try {
      defaultHeaders = JSON.parse(headersText.value);
    } catch {
      ElMessage.error("默认请求头不是合法 JSON");
      return;
    }
  }

  submitting.value = true;
  try {
    const uif = uifEnabled.value
      ? {
          enabled: true,
          field: form.uifField,
          prefix: form.uifPrefix || undefined,
          maxLen: form.uifMaxLen,
        }
      : null;
    await providerStore.saveProvider(
      {
        providerCode: form.providerCode,
        displayName: form.displayName,
        apiBaseUrl: form.apiBaseUrl,
        protocolType: form.protocolType,
        authType: form.authType,
        defaultHeaders,
        healthCheckEnabled: form.healthCheckEnabled,
        userIdentityForward: uif,
        sortOrder: form.sortOrder,
        remark: form.remark || null,
        status: form.status,
      },
      props.provider?.id
    );
    ElMessage.success("保存成功");
    emit("update:modelValue", false);
  } finally {
    submitting.value = false;
  }
}

// ==================== API Key ====================

const keyDialog = reactive({
  visible: false,
  mode: "create" as "create" | "edit",
  keyId: 0,
});
const emptyKeyForm = () => ({
  name: "",
  key: "",
  priority: 0,
  weight: 1,
  dailyQuota: 0,
  rpmLimit: 0,
  expiresAt: "",
});
const keyForm = reactive(emptyKeyForm());

const keyRules = {
  name: [{ required: true, message: "名称不能为空", trigger: "blur" }],
  key: [{ required: true, message: "Key 明文不能为空", trigger: "blur" }],
};

watch(keyDialog, ({ mode }) => {
  if (mode === "create") {
    Object.assign(keyForm, emptyKeyForm());
    keyDialog.visible = true;
  }
});

function openKeyEdit(row: ProviderKeyVO) {
  keyDialog.mode = "edit";
  keyDialog.keyId = row.id;
  keyDialog.visible = true;
  Object.assign(keyForm, emptyKeyForm(), {
    priority: row.priority,
    weight: row.weight,
    dailyQuota: row.dailyQuota ?? 0,
    rpmLimit: row.rpmLimit ?? 0,
    expiresAt: row.expiresAt ?? "",
  });
}

async function submitKey() {
  if (keyDialog.mode === "create") {
    await keyFormRef.value.validate();
  }
  const providerId = props.provider!.id;
  keySubmitting.value = true;
  try {
    // 日限额/RPM 前端用 0 表示不限，与后端 null 语义对齐
    const formPayload = {
      priority: keyForm.priority,
      weight: keyForm.weight,
      dailyQuota: keyForm.dailyQuota > 0 ? keyForm.dailyQuota : null,
      rpmLimit: keyForm.rpmLimit > 0 ? keyForm.rpmLimit : null,
      expiresAt: keyForm.expiresAt || null,
    };
    if (keyDialog.mode === "create") {
      await providerStore.createKey(providerId, { ...keyForm, ...formPayload });
      ElMessage.success("Key 已创建");
    } else {
      await providerStore.updateKey(providerId, keyDialog.keyId, formPayload);
      ElMessage.success("Key 已更新");
    }
    keyDialog.visible = false;
  } finally {
    keySubmitting.value = false;
  }
}

async function handleKeyStatusChange(row: ProviderKeyVO) {
  try {
    await providerStore.updateKey(props.provider!.id, row.id, {
      status: row.status,
    });
  } catch {
    row.status = row.status === 1 ? 0 : 1;
  }
}

async function handleKeyDelete(row: ProviderKeyVO) {
  await ElMessageBox.confirm(`确认删除 Key「${row.name}」？`, "删除确认", {
    type: "warning",
  });
  await providerStore.deleteKey(props.provider!.id, row.id);
  ElMessage.success("删除成功");
}
</script>
