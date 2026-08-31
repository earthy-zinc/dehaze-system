<!-- 成本单价管理：模型-供应商价格版本与档位明细维护 -->
<template>
  <div>
    <div class="flex justify-between items-center mb-2">
      <el-input
        v-model="billingStore.costQuery.keyword"
        clearable
        placeholder="模型标识"
        style="width: 200px"
        @keyup.enter="handleQuery"
      />
      <el-button
        v-hasPerm="['ai:billing:cost']"
        type="primary"
        @click="openDialog(null)"
      >
        <el-icon><Plus /></el-icon>新增成本版本
      </el-button>
    </div>

    <el-table
      v-loading="billingStore.costLoading"
      :data="billingStore.costVersions"
      size="small"
    >
      <el-table-column label="模型" prop="modelId" min-width="150" />
      <el-table-column label="供应商" width="120">
        <template #default="{ row }">{{
          providerName(row.providerId)
        }}</template>
      </el-table-column>
      <el-table-column
        label="版本"
        prop="priceVersion"
        width="70"
        align="center"
      />
      <el-table-column label="币种" prop="currency" width="70" align="center" />
      <el-table-column label="生效时间" prop="effectiveFrom" width="160" />
      <el-table-column label="失效时间" width="160">
        <template #default="{ row }">{{ row.effectiveTo ?? "长期" }}</template>
      </el-table-column>
      <el-table-column label="状态" width="80" align="center">
        <template #default="{ row }">
          <el-tag :type="row.status === 1 ? 'success' : 'info'" size="small">
            {{ row.status === 1 ? "启用" : "停用" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="档位明细" min-width="220">
        <template #default="{ row }">
          <div v-for="(d, i) in row.details" :key="i" class="text-xs leading-5">
            {{ d.tokenType }}/{{ d.timeSlot }} [{{ d.minTokens ?? 0 }}~{{
              d.maxTokens ?? "∞"
            }}]：{{ d.unitPrice }} 元/百万token
          </div>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="120" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            v-hasPerm="['ai:billing:cost']"
            link
            type="primary"
            size="small"
            @click="openDialog(row as ModelCostVO)"
          >
            编辑
          </el-button>
          <el-button
            v-hasPerm="['ai:billing:cost']"
            link
            type="danger"
            size="small"
            @click="handleDelete(row as ModelCostVO)"
          >
            停用
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-if="billingStore.costTotal > (billingStore.costQuery.pageSize ?? 10)"
      v-model:limit="billingStore.costQuery.pageSize"
      v-model:page="billingStore.costQuery.pageNum"
      v-model:total="billingStore.costTotal"
      @pagination="billingStore.fetchCostVersions()"
    />

    <!-- 成本版本表单弹窗：保存即生成新版本 -->
    <el-dialog
      v-model="dialogVisible"
      :title="editingId ? `编辑成本版本 - ${form.modelId}` : '新增成本版本'"
      width="820px"
      destroy-on-close
    >
      <el-form ref="formRef" :model="form" :rules="rules" label-width="100px">
        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="模型" prop="modelId">
              <el-select
                v-model="form.modelId"
                filterable
                placeholder="模型标识"
              >
                <el-option
                  v-for="m in modelOptions"
                  :key="m.modelId"
                  :label="`${m.displayName} (${m.modelId})`"
                  :value="m.modelId"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="供应商" prop="providerId">
              <el-select v-model="form.providerId" placeholder="供应商">
                <el-option
                  v-for="p in providerOptions"
                  :key="p.id"
                  :label="p.displayName"
                  :value="p.id"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="币种">
              <el-select v-model="form.currency">
                <el-option label="人民币 CNY" value="CNY" />
                <el-option label="美元 USD" value="USD" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="生效时间">
              <el-date-picker
                v-model="form.effectiveFrom"
                type="datetime"
                value-format="YYYY-MM-DD HH:mm:ss"
                placeholder="开始生效"
              />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="失效时间">
              <el-date-picker
                v-model="form.effectiveTo"
                type="datetime"
                value-format="YYYY-MM-DD HH:mm:ss"
                placeholder="空为长期有效"
              />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="状态">
              <el-switch
                v-model="form.status"
                :active-value="1"
                :inactive-value="0"
              />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>

      <div class="text-xs text-gray-400 mb-1">
        单价单位：元/百万token；不区分时段/分段的供应商仅配置最少档位即可
      </div>
      <el-table :data="detailRows" size="small">
        <el-table-column label="Token类型" width="130">
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
              <el-option label="idle 空闲" value="idle" />
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
              :precision="4"
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
              @click="detailRows.splice($index, 1)"
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
        @click="addDetailRow"
      >
        <el-icon><Plus /></el-icon>添加档位
      </el-button>

      <template #footer>
        <el-button type="primary" :loading="submitting" @click="submit"
          >保 存</el-button
        >
        <el-button @click="dialogVisible = false">取 消</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { Plus } from "@element-plus/icons-vue";
import {
  AiModelAPI,
  AiModelVO,
  AiProviderAPI,
  ModelCostDetailForm,
  ModelCostForm,
  ModelCostVO,
  ProviderVO,
} from "dehaze-sdk-js";
import { useAdminBillingStore } from "@/store/modules/adminBilling";

defineOptions({ name: "CostUnitPriceManage" });

const billingStore = useAdminBillingStore();

const dialogVisible = ref(false);
const editingId = ref<number>();
const submitting = ref(false);
const formRef = ref(ElForm);
const modelOptions = ref<AiModelVO[]>([]);
const providerOptions = ref<ProviderVO[]>([]);

const form = reactive<ModelCostForm>({
  modelId: "",
  providerId: undefined,
  currency: "CNY",
  effectiveFrom: undefined,
  effectiveTo: undefined,
  status: 1,
  details: [],
});
const detailRows = reactive<ModelCostDetailForm[]>([]);

const rules = {
  modelId: [{ required: true, message: "模型不能为空", trigger: "change" }],
  providerId: [
    { required: true, message: "供应商不能为空", trigger: "change" },
  ],
};

function providerName(providerId?: number) {
  return (
    providerOptions.value.find((p) => p.id === providerId)?.displayName ??
    providerId ??
    "-"
  );
}

function handleQuery() {
  billingStore.costQuery.pageNum = 1;
  billingStore.fetchCostVersions();
}

function addDetailRow() {
  detailRows.push({
    tokenType: "input",
    timeSlot: "peak",
    minTokens: 0,
    maxTokens: undefined,
    unitPrice: 0,
  });
}

async function openDialog(row: ModelCostVO | null) {
  if (modelOptions.value.length === 0 || providerOptions.value.length === 0) {
    const [modelPage, providerPage] = await Promise.all([
      AiModelAPI.listModels({ pageNum: 1, pageSize: 100 }),
      AiProviderAPI.listProviders({ pageNum: 1, pageSize: 100 }),
    ]);
    modelOptions.value = modelPage.list ?? [];
    providerOptions.value = providerPage.list ?? [];
  }
  editingId.value = row?.id;
  Object.assign(form, {
    modelId: row?.modelId ?? "",
    providerId: row?.providerId,
    currency: row?.currency ?? "CNY",
    effectiveFrom: row?.effectiveFrom,
    effectiveTo: row?.effectiveTo,
    status: row?.status ?? 1,
  });
  detailRows.splice(
    0,
    detailRows.length,
    ...(row?.details ?? []).map((d) => ({ ...d }))
  );
  if (detailRows.length === 0) addDetailRow();
  dialogVisible.value = true;
}

async function submit() {
  await formRef.value.validate();
  if (detailRows.length === 0) {
    ElMessage.warning("请至少配置一条成本档位");
    return;
  }
  submitting.value = true;
  try {
    await billingStore.saveCostVersion(
      { ...form, details: detailRows.map((row) => ({ ...row })) },
      editingId.value
    );
    ElMessage.success("成本版本已保存");
    dialogVisible.value = false;
  } finally {
    submitting.value = false;
  }
}

async function handleDelete(row: ModelCostVO) {
  await ElMessageBox.confirm(
    `确认停用「${row.modelId}」成本版本 v${row.priceVersion}？后续调用将按其他生效版本核算。`,
    "停用确认",
    { type: "warning" }
  );
  await billingStore.deleteCostVersion(row.id);
  ElMessage.success("已停用");
}

onMounted(() => {
  billingStore.fetchCostVersions();
});
</script>
