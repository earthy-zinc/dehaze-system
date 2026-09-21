<template>
  <div class="app-container">
    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <span
            >API Key
            用于脚本调用、定时任务等机器对机器场景，明文仅在创建时展示一次</span
          >
          <el-button type="success" @click="openCreateDialog">
            <el-icon><Plus /></el-icon>创建 API Key
          </el-button>
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="keyList"
        border
        highlight-current-row
      >
        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column
          label="名称"
          prop="name"
          min-width="160"
          show-overflow-tooltip
        />
        <el-table-column
          label="Key 前缀"
          prop="keyPrefix"
          width="160"
          align="center"
        />
        <el-table-column label="状态" width="90" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.status === 1 ? 'success' : 'danger'"
              size="small"
              effect="plain"
            >
              {{ scope.row.status === 1 ? "启用" : "禁用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="过期时间"
          prop="expiresAt"
          width="170"
          align="center"
        >
          <template #default="scope">
            {{ scope.row.expiresAt || "永不过期" }}
          </template>
        </el-table-column>
        <el-table-column
          label="最近使用"
          prop="lastUsedAt"
          width="170"
          align="center"
        >
          <template #default="scope">
            {{ scope.row.lastUsedAt || "-" }}
          </template>
        </el-table-column>
        <el-table-column
          label="创建时间"
          prop="createTime"
          width="170"
          align="center"
        />
        <el-table-column fixed="right" label="操作" width="100" align="center">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="danger"
              @click="handleDelete(scope.row as ApiKeyVO)"
            >
              <el-icon><Delete /></el-icon>删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="createVisible"
      title="创建 API Key"
      width="520px"
      append-to-body
    >
      <el-form
        ref="formRef"
        :model="formData"
        :rules="rules"
        label-width="100px"
      >
        <el-form-item label="名称" prop="name">
          <el-input
            v-model="formData.name"
            placeholder="用于标识 Key 用途"
            maxlength="64"
          />
        </el-form-item>
        <el-form-item label="过期时间" prop="expiresAt">
          <el-date-picker
            v-model="formData.expiresAt"
            type="datetime"
            format="YYYY-MM-DD HH:mm:ss"
            value-format="YYYY-MM-DD HH:mm:ss"
            placeholder="留空则永不过期"
            style="width: 100%"
          />
        </el-form-item>

        <el-collapse v-model="advancedOpen">
          <el-collapse-item name="advanced" title="高级选项">
            <el-form-item label="日配额" prop="dailyQuota">
              <el-input-number
                v-model="formData.dailyQuota"
                :min="0"
                :step="100"
                controls-position="right"
                placeholder="不限"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="月配额" prop="monthlyQuota">
              <el-input-number
                v-model="formData.monthlyQuota"
                :min="0"
                :step="100"
                controls-position="right"
                placeholder="不限"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="每分钟限速" prop="rpmLimit">
              <el-input-number
                v-model="formData.rpmLimit"
                :min="0"
                :step="10"
                controls-position="right"
                placeholder="不限"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="模型白名单" prop="modelWhitelist">
              <el-input
                v-model="modelWhitelistText"
                type="textarea"
                :rows="3"
                placeholder="每行一个模型标识，也可用逗号分隔；留空表示不限制"
              />
            </el-form-item>
          </el-collapse-item>
        </el-collapse>
      </el-form>
      <template #footer>
        <el-button type="primary" @click="handleCreate">确 定</el-button>
        <el-button @click="createVisible = false">取 消</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="revealVisible"
      title="API Key 明文"
      width="560px"
      append-to-body
    >
      <el-alert
        type="warning"
        :closable="false"
        show-icon
        title="明文仅此一次展示，请立即复制并妥善保存，关闭后无法再次查看"
      />
      <div class="reveal-row">
        <el-input :model-value="revealedKey" readonly>
          <template #append>
            <el-button @click="copyKey">
              <el-icon><CopyDocument /></el-icon>复制
            </el-button>
          </template>
        </el-input>
      </div>
      <template #footer>
        <el-button type="primary" @click="revealVisible = false"
          >我已保存</el-button
        >
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ApiKeyAPI, ApiKeyCreateForm, ApiKeyVO } from "dehaze-sdk-js";
import { CopyDocument, Delete, Plus } from "@element-plus/icons-vue";
import type { FormInstance } from "element-plus";

defineOptions({ name: "SystemApiKey" });

const loading = ref(false);
const keyList = ref<ApiKeyVO[]>([]);

const formRef = ref<FormInstance>();
const createVisible = ref(false);
const revealVisible = ref(false);
const revealedKey = ref("");

const advancedOpen = ref<string[]>([]);
const modelWhitelistText = ref("");

const formData = reactive<ApiKeyCreateForm>({ name: "" });
const rules = reactive({
  name: [{ required: true, message: "请输入 Key 名称", trigger: "blur" }],
  dailyQuota: [{ validator: validateNonNegative, trigger: "change" }],
  monthlyQuota: [{ validator: validateNonNegative, trigger: "change" }],
  rpmLimit: [{ validator: validateNonNegative, trigger: "change" }],
});

function validateNonNegative(
  _rule: unknown,
  value: number | undefined,
  callback: (error?: Error) => void
) {
  if (value === undefined || value === null || value >= 0) {
    callback();
  } else {
    callback(new Error("不能为负数"));
  }
}

function loadKeys() {
  loading.value = true;
  ApiKeyAPI.list()
    .then((data) => {
      keyList.value = data;
    })
    .finally(() => {
      loading.value = false;
    });
}

function openCreateDialog() {
  formData.name = "";
  formData.expiresAt = undefined;
  formData.dailyQuota = undefined;
  formData.monthlyQuota = undefined;
  formData.rpmLimit = undefined;
  formData.modelWhitelist = undefined;
  modelWhitelistText.value = "";
  advancedOpen.value = [];
  formRef.value?.clearValidate();
  createVisible.value = true;
}

function handleCreate() {
  formRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    const whitelist = modelWhitelistText.value
      .split(/[\n,，]/)
      .map((item) => item.trim())
      .filter(Boolean);
    ApiKeyAPI.create({
      ...formData,
      modelWhitelist: whitelist.length > 0 ? whitelist : undefined,
    }).then((data) => {
      createVisible.value = false;
      revealedKey.value = data.apiKey || "";
      revealVisible.value = true;
      loadKeys();
    });
  });
}

function copyKey() {
  navigator.clipboard.writeText(revealedKey.value).then(() => {
    ElMessage.success("已复制到剪贴板");
  });
}

function handleDelete(row: ApiKeyVO) {
  ElMessageBox.confirm(
    `确定删除 API Key「${row.name}」吗？删除后立即失效且不可恢复。`,
    "提示",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
      lockScroll: false,
    }
  )
    .then(() => ApiKeyAPI.delete(row.id))
    .then(() => {
      ElMessage.success("删除成功");
      loadKeys();
    })
    .catch(() => {});
}

onMounted(() => {
  loadKeys();
});
</script>

<style lang="scss" scoped>
.reveal-row {
  margin-top: 16px;
}
</style>
