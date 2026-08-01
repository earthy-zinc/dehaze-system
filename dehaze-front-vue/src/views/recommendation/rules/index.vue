<template>
  <div class="app-container">
    <el-card shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <span class="title">推荐规则管理</span>
          <el-button type="primary" @click="openDialog()">
            <el-icon><Plus /></el-icon>新增规则
          </el-button>
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="tableData as RecommendationRule[]"
        border
        stripe
      >
        <el-table-column label="规则名称" prop="ruleName" min-width="160" />
        <el-table-column
          label="场景类型"
          prop="sceneType"
          width="120"
          align="center"
        >
          <template #default="scope">
            <el-tag>{{
              getSceneTypeLabel((scope.row as RecommendationRule).sceneType)
            }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="关联算法" min-width="240">
          <template #default="scope">
            <el-tag
              v-for="(alg, idx) in getAlgoNames(
                (scope.row as RecommendationRule).algorithmIds
              )"
              :key="idx"
              size="small"
              style="margin: 2px"
            >
              {{ alg }}
            </el-tag>
            <span v-if="!(scope.row as RecommendationRule).algorithmIds?.length"
              >-</span
            >
          </template>
        </el-table-column>
        <el-table-column label="权重" prop="weight" width="100" align="center">
          <template #default="scope">
            <el-input-number
              v-model="(scope.row as RecommendationRule).weight"
              :min="0"
              :max="100"
              :step="5"
              size="small"
              style="width: 70px"
              @change="handleWeightChange(scope.row as RecommendationRule)"
            />
          </template>
        </el-table-column>
        <el-table-column label="启用状态" width="100" align="center">
          <template #default="scope">
            <el-switch
              v-model="(scope.row as RecommendationRule).enabled"
              @change="handleEnabledChange(scope.row as RecommendationRule)"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center" fixed="right">
          <template #default="scope">
            <el-button
              link
              type="primary"
              size="small"
              @click="openDialog(scope.row as RecommendationRule)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              link
              type="danger"
              size="small"
              @click="handleDelete(scope.row as RecommendationRule)"
            >
              <el-icon><Delete /></el-icon>删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Edit/Add Dialog -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.isEdit ? '编辑推荐规则' : '新增推荐规则'"
      width="600px"
      append-to-body
      destroy-on-close
      @close="closeDialog"
    >
      <el-form
        ref="formRef"
        :model="formData"
        :rules="rules"
        label-width="100px"
      >
        <el-form-item label="规则名称" prop="ruleName">
          <el-input v-model="formData.ruleName" placeholder="请输入规则名称" />
        </el-form-item>

        <el-form-item label="场景类型" prop="sceneType">
          <el-select
            v-model="formData.sceneType"
            placeholder="请选择场景类型"
            style="width: 100%"
          >
            <el-option label="城市" value="urban" />
            <el-option label="风景" value="landscape" />
            <el-option label="建筑" value="building" />
            <el-option label="夜景" value="night" />
            <el-option label="逆光" value="backlight" />
            <el-option label="室内" value="indoor" />
          </el-select>
        </el-form-item>

        <el-form-item label="关联算法" prop="algorithmIds">
          <el-select
            v-model="formData.algorithmIds"
            multiple
            filterable
            placeholder="请选择关联算法"
            style="width: 100%"
          >
            <el-option
              v-for="item in algorithmOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="权重" prop="weight">
          <el-slider
            v-model="formData.weight"
            :min="0"
            :max="100"
            :step="5"
            show-tooltip
          />
          <div style="margin-top: 4px; color: #909399; text-align: center">
            当前权重：{{ formData.weight }}
          </div>
        </el-form-item>

        <el-form-item label="启用状态">
          <el-switch v-model="formData.enabled" />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="closeDialog">取 消</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmit"
          >确 定</el-button
        >
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from "vue";
import {
  RecommendationAPI,
  RecommendationRule,
  AlgorithmAPI,
  OptionType,
} from "dehaze-sdk-js";
import { Plus, Edit, Delete } from "@element-plus/icons-vue";

defineOptions({ name: "RecommendationRules" });

const loading = ref(false);
const submitting = ref(false);
const tableData = ref<RecommendationRule[]>([]);
const algorithmOptions = ref<OptionType[]>([]);
// Local map for algorithm id -> name resolution
const algoNameMap = ref<Record<number, string>>({});

const dialog = reactive({
  visible: false,
  isEdit: false,
});

const formRef = ref<any>(null);
const formData = reactive<RecommendationRule>({
  ruleName: "",
  sceneType: "urban",
  algorithmIds: [],
  weight: 50,
  enabled: true,
});

const rules = {
  ruleName: [{ required: true, message: "请输入规则名称", trigger: "blur" }],
  sceneType: [{ required: true, message: "请选择场景类型", trigger: "change" }],
  algorithmIds: [
    { required: true, message: "请至少选择一个关联算法", trigger: "change" },
  ],
};

function getSceneTypeLabel(sceneType: string): string {
  const map: Record<string, string> = {
    urban: "城市",
    landscape: "风景",
    building: "建筑",
    night: "夜景",
    backlight: "逆光",
    indoor: "室内",
  };
  return map[sceneType] || sceneType;
}

function getAlgoNames(ids?: number[]): string[] {
  if (!ids?.length) return [];
  return ids.map((id) => algoNameMap.value[id] || String(id));
}

async function loadData() {
  loading.value = true;
  try {
    tableData.value = await RecommendationAPI.getRules();
    // Fetch algorithm options to resolve algorithm names
    const algoList = await AlgorithmAPI.getOption();
    algorithmOptions.value = algoList;
    algoNameMap.value = {};
    algoList.forEach((a) => {
      algoNameMap.value[Number(a.value)] = a.label;
    });
  } catch {
    ElMessage.error("加载推荐规则失败");
  } finally {
    loading.value = false;
  }
}

function openDialog(rule?: RecommendationRule) {
  if (rule) {
    dialog.isEdit = true;
    Object.assign(formData, rule);
  } else {
    dialog.isEdit = false;
    ElMessage.warning("新增功能暂不可用，请使用编辑修改现有规则");
    return;
  }
  dialog.visible = true;
}

function closeDialog() {
  dialog.visible = false;
  dialog.isEdit = false;
  formRef.value?.resetFields();
}

async function handleSubmit() {
  if (!formRef.value) return;
  formRef.value.validate(async (valid: boolean) => {
    if (!valid) return;
    submitting.value = true;
    try {
      const id = await RecommendationAPI.updateRule(formData.id!, formData);
      if (id) {
        ElMessage.success("更新成功");
        closeDialog();
        loadData();
      }
    } catch {
      ElMessage.error("操作失败");
    } finally {
      submitting.value = false;
    }
  });
}

async function handleWeightChange(rule: RecommendationRule) {
  try {
    await RecommendationAPI.updateRule(rule.id!, { ...rule });
    ElMessage.success("权重已更新");
  } catch {
    ElMessage.error("权重更新失败");
    loadData();
  }
}

async function handleEnabledChange(rule: RecommendationRule) {
  try {
    await RecommendationAPI.updateRule(rule.id!, { ...rule });
  } catch {
    ElMessage.error("状态更新失败");
    loadData();
  }
}

async function handleDelete(rule: RecommendationRule) {
  ElMessageBox.confirm(`确认删除规则"${rule.ruleName}"吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(async () => {
      // Note: No deleteRule API defined in SDK, using update with disabled approach
      await RecommendationAPI.updateRule(rule.id!, { ...rule, enabled: false });
      ElMessage.success("已禁用该规则");
      loadData();
    })
    .catch(() => {});
}

onMounted(() => {
  loadData();
});
</script>

<style scoped>
.title {
  font-size: 18px;
  font-weight: 600;
}
</style>
