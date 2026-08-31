<!-- 语音引擎注册表管理（F-VS-005）：引擎/密钥/模型音色配置，需 voice:engine:manage 权限 -->
<script lang="ts" setup>
import { Plus, Refresh } from "@element-plus/icons-vue";
import type { FormInstance } from "element-plus";
import {
  VoiceModelCreateForm,
  VoiceModelUpdateForm,
  VoiceModelVO,
  VoiceProviderKeyCreateForm,
  VoiceProviderKeyUpdateForm,
  VoiceProviderKeyVO,
  VoiceProviderVO,
  VoiceAPI,
} from "dehaze-sdk-js";

const ENGINE_TYPES = [
  { value: "asr", label: "语音识别（ASR）" },
  { value: "tts", label: "语音合成（TTS）" },
];

const AUTH_TYPES = [
  { value: "bearer", label: "Bearer Token" },
  { value: "x-api-key", label: "X-API-Key" },
  { value: "custom", label: "自定义（默认请求头）" },
];

const loading = ref(false);
const providers = ref<VoiceProviderVO[]>([]);
const total = ref(0);
const query = reactive({
  pageNum: 1,
  pageSize: 10,
  keyword: "",
  engineType: "",
});

async function fetchProviders() {
  loading.value = true;
  try {
    const page = await VoiceAPI.listProviders({
      pageNum: query.pageNum,
      pageSize: query.pageSize,
      keyword: query.keyword || undefined,
      engineType: query.engineType || undefined,
    });
    providers.value = page.list;
    total.value = page.total;
  } finally {
    loading.value = false;
  }
}

function handleQuery() {
  query.pageNum = 1;
  fetchProviders();
}

// ===== 引擎新增/编辑 =====

const dialog = reactive({ visible: false, isEdit: false, saving: false });
/** 编辑时记录原始行（provider_code/engine_type 不可变更） */
let editing: VoiceProviderVO | null = null;
const providerForm = reactive({
  providerCode: "",
  engineType: "asr",
  displayName: "",
  apiBaseUrl: "",
  authType: "bearer",
  headersText: "",
  isDefault: 0,
  status: 1,
  sortOrder: 0,
  healthCheckEnabled: 1,
  remark: "",
});

const providerFormRef = ref<FormInstance>();

function openCreate() {
  editing = null;
  Object.assign(providerForm, {
    providerCode: "",
    engineType: "asr",
    displayName: "",
    apiBaseUrl: "",
    authType: "bearer",
    headersText: "",
    isDefault: 0,
    status: 1,
    sortOrder: 0,
    healthCheckEnabled: 1,
    remark: "",
  });
  dialog.isEdit = false;
  dialog.visible = true;
}

function openEdit(row: VoiceProviderVO) {
  editing = row;
  Object.assign(providerForm, {
    providerCode: row.providerCode,
    engineType: row.engineType,
    displayName: row.displayName,
    apiBaseUrl: row.apiBaseUrl ?? "",
    authType: row.authType,
    headersText: row.defaultHeaders
      ? JSON.stringify(row.defaultHeaders, null, 2)
      : "",
    isDefault: row.isDefault,
    status: row.status,
    sortOrder: row.sortOrder,
    healthCheckEnabled: row.healthCheckEnabled,
    remark: row.remark ?? "",
  });
  dialog.isEdit = true;
  dialog.visible = true;
}

/** JSON 文本编辑字段（默认请求头）统一解析：空文本置 null，非法 JSON 阻断提交 */
function parseJsonText(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      throw new Error("not an object");
    }
    return parsed;
  } catch {
    ElMessage.warning("默认请求头不是合法的 JSON 对象");
    throw new Error("invalid json");
  }
}

async function submitProvider() {
  await providerFormRef.value
    ?.validate()
    .catch(() => Promise.reject(new Error("invalid form")));
  const headers = parseJsonText(providerForm.headersText);
  dialog.saving = true;
  try {
    if (dialog.isEdit && editing) {
      await VoiceAPI.updateProvider(editing.id, {
        displayName: providerForm.displayName,
        apiBaseUrl: providerForm.apiBaseUrl || null,
        authType: providerForm.authType,
        defaultHeaders: headers,
        isDefault: providerForm.isDefault,
        status: providerForm.status,
        sortOrder: providerForm.sortOrder,
        healthCheckEnabled: providerForm.healthCheckEnabled,
        remark: providerForm.remark || null,
      });
      ElMessage.success("引擎已更新");
    } else {
      await VoiceAPI.createProvider({
        providerCode: providerForm.providerCode,
        engineType: providerForm.engineType,
        displayName: providerForm.displayName,
        apiBaseUrl: providerForm.apiBaseUrl || null,
        authType: providerForm.authType,
        defaultHeaders: headers,
        isDefault: providerForm.isDefault,
        status: providerForm.status,
        sortOrder: providerForm.sortOrder,
        healthCheckEnabled: providerForm.healthCheckEnabled,
        remark: providerForm.remark || null,
      });
      ElMessage.success("引擎已创建");
    }
    dialog.visible = false;
    await fetchProviders();
  } finally {
    dialog.saving = false;
  }
}

async function setDefault(row: VoiceProviderVO) {
  await VoiceAPI.updateProvider(row.id, { isDefault: 1 });
  ElMessage.success(`已将「${row.displayName}」设为默认引擎`);
  await fetchProviders();
}

async function handleStatusChange(row: VoiceProviderVO) {
  try {
    await VoiceAPI.updateProvider(row.id, { status: row.status });
    ElMessage.success(row.status === 1 ? "已启用" : "已停用");
  } catch {
    row.status = row.status === 1 ? 0 : 1;
  }
}

async function handleDelete(row: VoiceProviderVO) {
  await ElMessageBox.confirm(
    `确认删除引擎「${row.displayName}」？引擎编码 ${row.providerCode} 删除后不可复用。`,
    "删除确认",
    { type: "warning" }
  );
  await VoiceAPI.deleteProvider(row.id);
  ElMessage.success("删除成功");
  await fetchProviders();
}

const testingId = ref<number | null>(null);

async function handleTest(row: VoiceProviderVO) {
  testingId.value = row.id;
  try {
    const result = await VoiceAPI.testProviderConnection(row.id);
    ElMessageBox.alert(
      `<pre style="max-height:300px;overflow:auto">${JSON.stringify(result, null, 2)}</pre>`,
      `连通性测试结果 - ${row.displayName}`,
      { dangerouslyUseHTMLString: true }
    );
  } finally {
    testingId.value = null;
  }
}

// ===== 密钥管理 =====

const keysDrawer = reactive({ visible: false, loading: false });
const keysProvider = ref<VoiceProviderVO | null>(null);
const keys = ref<VoiceProviderKeyVO[]>([]);

const keyDialog = reactive({ visible: false, isEdit: false, saving: false });
let editingKey: VoiceProviderKeyVO | null = null;
const keyForm = reactive({
  name: "",
  key: "",
  status: 1,
  priority: 0,
  weight: 1,
  dailyQuota: undefined as number | undefined,
  rpmLimit: undefined as number | undefined,
  expiresAt: "",
});

async function openKeys(row: VoiceProviderVO) {
  keysProvider.value = row;
  keysDrawer.visible = true;
  await fetchKeys();
}

async function fetchKeys() {
  if (!keysProvider.value) return;
  keysDrawer.loading = true;
  try {
    keys.value = await VoiceAPI.listProviderKeys(keysProvider.value.id);
  } finally {
    keysDrawer.loading = false;
  }
}

function openKeyCreate() {
  editingKey = null;
  Object.assign(keyForm, {
    name: "",
    key: "",
    status: 1,
    priority: 0,
    weight: 1,
    dailyQuota: undefined,
    rpmLimit: undefined,
    expiresAt: "",
  });
  keyDialog.isEdit = false;
  keyDialog.visible = true;
}

function openKeyEdit(row: VoiceProviderKeyVO) {
  editingKey = row;
  Object.assign(keyForm, {
    name: row.name,
    key: "",
    status: row.status,
    priority: row.priority,
    weight: row.weight,
    dailyQuota: row.dailyQuota ?? undefined,
    rpmLimit: row.rpmLimit ?? undefined,
    expiresAt: row.expiresAt ?? "",
  });
  keyDialog.isEdit = true;
  keyDialog.visible = true;
}

async function submitKey() {
  keyDialog.saving = true;
  try {
    if (keyDialog.isEdit && editingKey && keysProvider.value) {
      const form: VoiceProviderKeyUpdateForm = {
        name: keyForm.name,
        status: keyForm.status,
        priority: keyForm.priority,
        weight: keyForm.weight,
        dailyQuota: keyForm.dailyQuota ?? null,
        rpmLimit: keyForm.rpmLimit ?? null,
        expiresAt: keyForm.expiresAt || null,
      };
      await VoiceAPI.updateProviderKey(
        keysProvider.value.id,
        editingKey.id,
        form
      );
      ElMessage.success("API Key 已更新");
    } else if (keysProvider.value) {
      const form: VoiceProviderKeyCreateForm = {
        name: keyForm.name,
        key: keyForm.key,
        status: keyForm.status,
        priority: keyForm.priority,
        weight: keyForm.weight,
        dailyQuota: keyForm.dailyQuota ?? null,
        rpmLimit: keyForm.rpmLimit ?? null,
        expiresAt: keyForm.expiresAt || null,
      };
      await VoiceAPI.createProviderKey(keysProvider.value.id, form);
      ElMessage.success("API Key 已创建");
    }
    keyDialog.visible = false;
    await fetchKeys();
  } finally {
    keyDialog.saving = false;
  }
}

async function handleKeyStatusChange(row: VoiceProviderKeyVO) {
  if (!keysProvider.value) return;
  try {
    await VoiceAPI.updateProviderKey(keysProvider.value.id, row.id, {
      status: row.status,
    });
  } catch {
    row.status = row.status === 1 ? 0 : 1;
  }
}

async function handleKeyDelete(row: VoiceProviderKeyVO) {
  await ElMessageBox.confirm(
    `确认删除 API Key「${row.name}」？该操作为物理删除，不可恢复。`,
    "删除确认",
    { type: "warning" }
  );
  if (!keysProvider.value) return;
  await VoiceAPI.deleteProviderKey(keysProvider.value.id, row.id);
  ElMessage.success("删除成功");
  await fetchKeys();
}

// ===== 模型/音色管理 =====

const modelsDrawer = reactive({ visible: false, loading: false });
const modelsProvider = ref<VoiceProviderVO | null>(null);
const models = ref<VoiceModelVO[]>([]);

const modelDialog = reactive({ visible: false, isEdit: false, saving: false });
let editingModel: VoiceModelVO | null = null;
const modelForm = reactive({
  modelId: "",
  modelType: "stream",
  displayName: "",
  paramsText: "",
  status: 1,
});

const MODEL_TYPES: Record<string, { value: string; label: string }[]> = {
  asr: [
    { value: "stream", label: "流式识别" },
    { value: "offline", label: "离线识别" },
  ],
  tts: [{ value: "voice", label: "音色" }],
};

async function openModels(row: VoiceProviderVO) {
  modelsProvider.value = row;
  modelsDrawer.visible = true;
  await fetchModels();
}

async function fetchModels() {
  if (!modelsProvider.value) return;
  modelsDrawer.loading = true;
  try {
    const all = await VoiceAPI.listVoiceModels({
      engineType: modelsProvider.value.engineType,
    });
    models.value = all.filter((m) => m.providerId === modelsProvider.value!.id);
  } finally {
    modelsDrawer.loading = false;
  }
}

function openModelCreate() {
  editingModel = null;
  Object.assign(modelForm, {
    modelId: "",
    modelType:
      MODEL_TYPES[modelsProvider.value?.engineType ?? "asr"]![0]!.value,
    displayName: "",
    paramsText: "",
    status: 1,
  });
  modelDialog.isEdit = false;
  modelDialog.visible = true;
}

function openModelEdit(row: VoiceModelVO) {
  editingModel = row;
  Object.assign(modelForm, {
    modelId: row.modelId,
    modelType: row.modelType,
    displayName: row.displayName,
    paramsText: row.params ? JSON.stringify(row.params, null, 2) : "",
    status: row.status,
  });
  modelDialog.isEdit = true;
  modelDialog.visible = true;
}

async function submitModel() {
  const params = parseJsonText(modelForm.paramsText);
  modelDialog.saving = true;
  try {
    if (modelDialog.isEdit && editingModel) {
      const form: VoiceModelUpdateForm = {
        displayName: modelForm.displayName,
        params,
        status: modelForm.status,
      };
      await VoiceAPI.updateVoiceModel(editingModel.id, form);
      ElMessage.success("模型/音色已更新");
    } else if (modelsProvider.value) {
      const form: VoiceModelCreateForm = {
        providerId: modelsProvider.value.id,
        modelId: modelForm.modelId,
        engineType: modelsProvider.value.engineType,
        modelType: modelForm.modelType,
        displayName: modelForm.displayName,
        params,
        status: modelForm.status,
      };
      await VoiceAPI.createVoiceModel(form);
      ElMessage.success("模型/音色已创建");
    }
    modelDialog.visible = false;
    await fetchModels();
  } finally {
    modelDialog.saving = false;
  }
}

async function handleModelDelete(row: VoiceModelVO) {
  await ElMessageBox.confirm(
    `确认删除模型/音色「${row.displayName}」？业务编码 ${row.modelId} 删除后保留占用，不可复用。`,
    "删除确认",
    { type: "warning" }
  );
  await VoiceAPI.deleteVoiceModel(row.id);
  ElMessage.success("删除成功");
  await fetchModels();
}

onMounted(() => {
  fetchProviders();
});
</script>

<template>
  <div>
    <div class="flex justify-between mb-4">
      <div class="flex items-center gap-2">
        <el-select
          v-model="query.engineType"
          placeholder="能力类型"
          clearable
          class="!w-36"
          @change="handleQuery"
        >
          <el-option
            v-for="item in ENGINE_TYPES"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-input
          v-model="query.keyword"
          placeholder="显示名称/引擎编码"
          clearable
          class="!w-56"
          @keyup.enter="handleQuery"
          @clear="handleQuery"
        />
        <el-button @click="handleQuery">查询</el-button>
      </div>
      <div class="flex gap-2">
        <el-button @click="fetchProviders()">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
        <el-button type="success" @click="openCreate">
          <el-icon><Plus /></el-icon>
          新增引擎
        </el-button>
      </div>
    </div>

    <el-table v-loading="loading" :data="providers" border>
      <el-table-column prop="providerCode" label="引擎编码" min-width="110" />
      <el-table-column prop="displayName" label="显示名称" min-width="150" />
      <el-table-column label="能力类型" width="100" align="center">
        <template #default="{ row }">
          <el-tag
            :type="row.engineType === 'asr' ? 'primary' : 'success'"
            size="small"
          >
            {{ row.engineType === "asr" ? "ASR" : "TTS" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        prop="authType"
        label="认证方式"
        width="110"
        align="center"
      />
      <el-table-column label="默认" width="80" align="center">
        <template #default="{ row }">
          <el-tag v-if="row.isDefault === 1" type="warning" size="small"
            >默认</el-tag
          >
        </template>
      </el-table-column>
      <el-table-column label="状态" width="80" align="center">
        <template #default="{ row }">
          <el-switch
            v-model="row.status"
            :active-value="1"
            :inactive-value="0"
            @change="handleStatusChange(row as VoiceProviderVO)"
          />
        </template>
      </el-table-column>
      <el-table-column
        prop="sortOrder"
        label="排序"
        width="70"
        align="center"
      />
      <el-table-column label="操作" width="320" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            v-if="row.isDefault !== 1"
            link
            type="warning"
            size="small"
            @click="setDefault(row as VoiceProviderVO)"
          >
            设为默认
          </el-button>
          <el-button
            link
            type="primary"
            size="small"
            @click="openEdit(row as VoiceProviderVO)"
          >
            编辑
          </el-button>
          <el-button
            link
            type="primary"
            size="small"
            @click="openKeys(row as VoiceProviderVO)"
          >
            密钥
          </el-button>
          <el-button
            link
            type="primary"
            size="small"
            @click="openModels(row as VoiceProviderVO)"
          >
            模型/音色
          </el-button>
          <el-button
            link
            type="primary"
            size="small"
            :loading="testingId === row.id"
            @click="handleTest(row as VoiceProviderVO)"
          >
            连通测试
          </el-button>
          <el-button
            link
            type="danger"
            size="small"
            @click="handleDelete(row as VoiceProviderVO)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-if="total > 0"
      v-model:limit="query.pageSize"
      v-model:page="query.pageNum"
      v-model:total="total"
      @pagination="fetchProviders"
    />

    <!-- 引擎新增/编辑弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.isEdit ? '编辑引擎' : '新增引擎'"
      width="560px"
    >
      <el-form ref="providerFormRef" :model="providerForm" label-width="120px">
        <el-form-item label="引擎编码" required>
          <el-input
            v-model="providerForm.providerCode"
            :disabled="dialog.isEdit"
            maxlength="32"
            placeholder="如 local / aliyun（同能力类型下唯一）"
          />
        </el-form-item>
        <el-form-item label="能力类型" required>
          <el-radio-group
            v-model="providerForm.engineType"
            :disabled="dialog.isEdit"
          >
            <el-radio value="asr">ASR</el-radio>
            <el-radio value="tts">TTS</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="显示名称" required>
          <el-input v-model="providerForm.displayName" maxlength="128" />
        </el-form-item>
        <el-form-item label="API 基础地址">
          <el-input
            v-model="providerForm.apiBaseUrl"
            maxlength="512"
            placeholder="云端引擎必填，local 引擎留空"
          />
        </el-form-item>
        <el-form-item label="认证方式">
          <el-select v-model="providerForm.authType" class="!w-full">
            <el-option
              v-for="item in AUTH_TYPES"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="默认请求头">
          <el-input
            v-model="providerForm.headersText"
            type="textarea"
            :rows="4"
            placeholder='JSON 对象，如 {"X-Custom-Auth": "xxx"}'
          />
        </el-form-item>
        <el-form-item label="设为默认引擎">
          <el-switch
            v-model="providerForm.isDefault"
            :active-value="1"
            :inactive-value="0"
          />
          <span class="ml-2 text-xs text-gray-400"
            >同能力类型下仅一个默认引擎，后端自动清其他默认</span
          >
        </el-form-item>
        <el-form-item label="状态">
          <el-switch
            v-model="providerForm.status"
            :active-value="1"
            :inactive-value="0"
          />
        </el-form-item>
        <el-form-item label="排序序号">
          <el-input-number v-model="providerForm.sortOrder" :min="0" />
        </el-form-item>
        <el-form-item label="健康检查">
          <el-switch
            v-model="providerForm.healthCheckEnabled"
            :active-value="1"
            :inactive-value="0"
          />
        </el-form-item>
        <el-form-item label="备注">
          <el-input
            v-model="providerForm.remark"
            type="textarea"
            :rows="2"
            maxlength="512"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialog.visible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="dialog.saving"
          @click="submitProvider"
        >
          确定
        </el-button>
      </template>
    </el-dialog>

    <!-- 密钥管理抽屉 -->
    <el-drawer
      v-model="keysDrawer.visible"
      :title="`API Key 管理 - ${keysProvider?.displayName ?? ''}`"
      size="720px"
    >
      <div class="flex justify-end mb-3">
        <el-button type="success" size="small" @click="openKeyCreate">
          <el-icon><Plus /></el-icon>
          新增 Key
        </el-button>
      </div>
      <el-table v-loading="keysDrawer.loading" :data="keys" border size="small">
        <el-table-column prop="name" label="名称" min-width="120" />
        <el-table-column prop="keyPrefix" label="密钥前缀" min-width="140" />
        <el-table-column label="状态" width="70" align="center">
          <template #default="{ row }">
            <el-switch
              v-model="row.status"
              :active-value="1"
              :inactive-value="0"
              size="small"
              @change="handleKeyStatusChange(row as VoiceProviderKeyVO)"
            />
          </template>
        </el-table-column>
        <el-table-column
          prop="priority"
          label="优先级"
          width="70"
          align="center"
        />
        <el-table-column prop="weight" label="权重" width="70" align="center" />
        <el-table-column label="日限额" width="80" align="center">
          <template #default="{ row }">{{ row.dailyQuota ?? "不限" }}</template>
        </el-table-column>
        <el-table-column label="RPM" width="70" align="center">
          <template #default="{ row }">{{
            row.rpmLimit == null
              ? "不限"
              : row.rpmLimit === 0
                ? "不限"
                : row.rpmLimit
          }}</template>
        </el-table-column>
        <el-table-column prop="expiresAt" label="过期时间" width="160" />
        <el-table-column label="操作" width="110" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              link
              type="primary"
              size="small"
              @click="openKeyEdit(row as VoiceProviderKeyVO)"
            >
              编辑
            </el-button>
            <el-button
              link
              type="danger"
              size="small"
              @click="handleKeyDelete(row as VoiceProviderKeyVO)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-drawer>

    <!-- Key 新增/编辑弹窗 -->
    <el-dialog
      v-model="keyDialog.visible"
      :title="keyDialog.isEdit ? '编辑 API Key' : '新增 API Key'"
      width="480px"
      append-to-body
    >
      <el-form :model="keyForm" label-width="110px">
        <el-form-item label="名称" required>
          <el-input v-model="keyForm.name" maxlength="128" />
        </el-form-item>
        <el-form-item v-if="!keyDialog.isEdit" label="Key 明文" required>
          <el-input
            v-model="keyForm.key"
            show-password
            placeholder="提交后仅保留掩码前缀，不再显示"
          />
        </el-form-item>
        <el-form-item label="状态">
          <el-switch
            v-model="keyForm.status"
            :active-value="1"
            :inactive-value="0"
          />
        </el-form-item>
        <el-form-item label="优先级">
          <el-input-number v-model="keyForm.priority" :min="0" />
          <span class="ml-2 text-xs text-gray-400">数字越小越优先</span>
        </el-form-item>
        <el-form-item label="权重">
          <el-input-number v-model="keyForm.weight" :min="1" />
        </el-form-item>
        <el-form-item label="日调用上限">
          <el-input-number
            v-model="keyForm.dailyQuota"
            :min="1"
            placeholder="留空不限"
          />
        </el-form-item>
        <el-form-item label="RPM 上限">
          <el-input-number
            v-model="keyForm.rpmLimit"
            :min="0"
            placeholder="0 或留空不限"
          />
        </el-form-item>
        <el-form-item label="过期时间">
          <el-date-picker
            v-model="keyForm.expiresAt"
            type="datetime"
            value-format="YYYY-MM-DD HH:mm:ss"
            placeholder="留空永不过期"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="keyDialog.visible = false">取消</el-button>
        <el-button type="primary" :loading="keyDialog.saving" @click="submitKey"
          >确定</el-button
        >
      </template>
    </el-dialog>

    <!-- 模型/音色管理抽屉 -->
    <el-drawer
      v-model="modelsDrawer.visible"
      :title="`模型/音色管理 - ${modelsProvider?.displayName ?? ''}`"
      size="720px"
    >
      <div class="flex justify-end mb-3">
        <el-button type="success" size="small" @click="openModelCreate">
          <el-icon><Plus /></el-icon>
          新增模型/音色
        </el-button>
      </div>
      <el-table
        v-loading="modelsDrawer.loading"
        :data="models"
        border
        size="small"
      >
        <el-table-column prop="modelId" label="业务编码" min-width="120" />
        <el-table-column prop="displayName" label="显示名称" min-width="120" />
        <el-table-column
          prop="modelType"
          label="子类型"
          width="90"
          align="center"
        />
        <el-table-column label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.status === 1 ? 'success' : 'info'" size="small">
              {{ row.status === 1 ? "启用" : "禁用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="参数" min-width="160">
          <template #default="{ row }">
            <span class="text-xs text-gray-500">
              {{ row.params ? JSON.stringify(row.params) : "-" }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="140" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              link
              type="primary"
              size="small"
              @click="openModelEdit(row as VoiceModelVO)"
            >
              编辑
            </el-button>
            <el-button
              link
              type="danger"
              size="small"
              @click="handleModelDelete(row as VoiceModelVO)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-drawer>

    <!-- 模型/音色新增/编辑弹窗 -->
    <el-dialog
      v-model="modelDialog.visible"
      :title="modelDialog.isEdit ? '编辑模型/音色' : '新增模型/音色'"
      width="480px"
      append-to-body
    >
      <el-form :model="modelForm" label-width="110px">
        <el-form-item label="业务编码" required>
          <el-input
            v-model="modelForm.modelId"
            :disabled="modelDialog.isEdit"
            maxlength="64"
            placeholder="如 sensevoice / huayan（删除后保留占用不可复用）"
          />
        </el-form-item>
        <el-form-item label="子类型" required>
          <el-select
            v-model="modelForm.modelType"
            :disabled="modelDialog.isEdit"
          >
            <el-option
              v-for="item in MODEL_TYPES[modelsProvider?.engineType ?? 'asr']"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="显示名称" required>
          <el-input v-model="modelForm.displayName" maxlength="128" />
        </el-form-item>
        <el-form-item label="模型参数">
          <el-input
            v-model="modelForm.paramsText"
            type="textarea"
            :rows="4"
            placeholder='JSON 对象，如 {"sampleRate": 16000}'
          />
        </el-form-item>
        <el-form-item label="状态">
          <el-switch
            v-model="modelForm.status"
            :active-value="1"
            :inactive-value="0"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="modelDialog.visible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="modelDialog.saving"
          @click="submitModel"
        >
          确定
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>
