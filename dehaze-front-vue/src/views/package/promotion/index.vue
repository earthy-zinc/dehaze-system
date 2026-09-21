<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="活动名称" prop="name">
          <el-input
            v-model="queryParams.name"
            clearable
            placeholder="活动名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="类型" prop="type">
          <el-select
            v-model="queryParams.type"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option label="限时折扣" value="discount" />
            <el-option label="新用户专享" value="new_user" />
            <el-option label="节日促销" value="holiday" />
            <el-option label="满减活动" value="full_reduction" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-select
            v-model="queryParams.status"
            clearable
            placeholder="全部"
            style="width: 120px"
          >
            <el-option label="启用" :value="1" />
            <el-option label="禁用" :value="0" />
          </el-select>
        </el-form-item>
        <el-form-item label="活动时间">
          <el-date-picker
            v-model="dateRange"
            type="daterange"
            value-format="YYYY-MM-DD"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            @change="handleDateChange"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <el-icon><Search /></el-icon>搜索
          </el-button>
          <el-button @click="resetQuery">
            <el-icon><Refresh /></el-icon>重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <el-button
          v-hasPerm="['package:promotion:add']"
          type="success"
          @click="openDialog()"
        >
          <el-icon><Plus /></el-icon>新增活动
        </el-button>
      </template>

      <el-table
        v-loading="loading"
        :data="promotionList"
        border
        highlight-current-row
      >
        <el-table-column label="活动名称" prop="name" min-width="160" />
        <el-table-column label="类型" align="center" width="110">
          <template #default="scope">
            <el-tag
              :type="typeTag((scope.row as PromotionVO).type)"
              effect="plain"
            >
              {{ typeLabel((scope.row as PromotionVO).type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="活动时间" align="center" min-width="280">
          <template #default="scope">
            {{ (scope.row as PromotionVO).startTime }} ~
            {{ (scope.row as PromotionVO).endTime }}
          </template>
        </el-table-column>
        <el-table-column label="新用户专享" align="center" width="100">
          <template #default="scope">
            <el-tag
              :type="
                (scope.row as PromotionVO).newUserOnly === 1
                  ? 'warning'
                  : 'info'
              "
              effect="plain"
            >
              {{ (scope.row as PromotionVO).newUserOnly === 1 ? "是" : "否" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" align="center" width="90">
          <template #default="scope">
            <el-tag
              :type="
                (scope.row as PromotionVO).status === 1 ? 'success' : 'info'
              "
              effect="plain"
            >
              {{ (scope.row as PromotionVO).status === 1 ? "启用" : "禁用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" fixed="right" width="260" align="center">
          <template #default="scope">
            <el-button
              v-hasPerm="['package:promotion:edit']"
              link
              size="small"
              type="primary"
              @click="handleEdit(scope.row as PromotionVO)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-hasPerm="['package:promotion:edit']"
              link
              size="small"
              type="warning"
              @click="openLinkDialog(scope.row as PromotionVO)"
            >
              <el-icon><Link /></el-icon>关联套餐
            </el-button>
            <el-button
              v-hasPerm="['package:promotion:edit']"
              link
              size="small"
              :type="
                (scope.row as PromotionVO).status === 1 ? 'info' : 'success'
              "
              @click="handleToggleStatus(scope.row as PromotionVO)"
            >
              {{ (scope.row as PromotionVO).status === 1 ? "禁用" : "启用" }}
            </el-button>
            <el-button
              v-hasPerm="['package:promotion:delete']"
              link
              size="small"
              type="danger"
              @click="handleDelete(scope.row as PromotionVO)"
            >
              <el-icon><Delete /></el-icon>删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="total > 0"
        v-model:limit="queryParams.pageSize"
        v-model:page="queryParams.pageNum"
        v-model:total="total"
        @pagination="handleQuery"
      />
    </el-card>

    <!-- 活动表单弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="680px"
      @close="closeDialog"
    >
      <el-form
        ref="promotionFormRef"
        :model="formData"
        :rules="rules"
        label-width="120px"
      >
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="活动名称" prop="name">
              <el-input v-model="formData.name" placeholder="活动名称" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="活动类型" prop="type">
              <el-select v-model="formData.type" style="width: 100%">
                <el-option label="限时折扣" value="discount" />
                <el-option label="新用户专享" value="new_user" />
                <el-option label="节日促销" value="holiday" />
                <el-option label="满减活动" value="full_reduction" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="开始时间" prop="startTime">
              <el-date-picker
                v-model="formData.startTime"
                type="datetime"
                value-format="YYYY-MM-DD HH:mm:ss"
                placeholder="开始时间"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="结束时间" prop="endTime">
              <el-date-picker
                v-model="formData.endTime"
                type="datetime"
                value-format="YYYY-MM-DD HH:mm:ss"
                placeholder="结束时间"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="折扣方式" prop="discountType">
              <el-select v-model="formData.discountType" style="width: 100%">
                <el-option label="百分比折扣" value="percent" />
                <el-option label="固定减免" value="fixed" />
                <el-option
                  v-if="formData.type === 'full_reduction'"
                  label="满减档位"
                  value="full_reduction"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col v-if="formData.discountType === 'percent'" :span="12">
            <el-form-item label="折扣值(%)" prop="discountPercent">
              <el-input-number
                v-model="formData.discountPercent"
                :min="1"
                :max="99"
                :precision="0"
                controls-position="right"
                style="width: 100%"
              />
              <div class="form-tip">减免原价的百分比，如 20 表示 8 折</div>
            </el-form-item>
          </el-col>
          <el-col v-if="formData.discountType === 'fixed'" :span="12">
            <el-form-item label="减免金额(元)" prop="discountFixedYuan">
              <el-input-number
                v-model="formData.discountFixedYuan"
                :min="0"
                :precision="2"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="新用户专享" prop="newUserOnly">
              <el-switch
                v-model="formData.newUserOnly"
                :active-value="1"
                :inactive-value="0"
              />
            </el-form-item>
          </el-col>
          <el-col :span="24">
            <el-form-item label="活动描述" prop="description">
              <el-input
                v-model="formData.description"
                type="textarea"
                :rows="2"
                placeholder="活动描述"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <template v-if="formData.discountType === 'full_reduction'">
          <el-divider content-position="left">满减档位（元）</el-divider>
          <div
            v-for="(tier, index) in formData.tiers"
            :key="index"
            class="tier-row"
          >
            <el-input-number
              v-model="tier.threshold"
              :min="0"
              :precision="2"
              controls-position="right"
              placeholder="满足金额"
            />
            <span class="tier-sep">减</span>
            <el-input-number
              v-model="tier.faceValue"
              :min="0"
              :precision="2"
              controls-position="right"
              placeholder="减免金额"
            />
            <el-button
              type="danger"
              link
              @click="formData.tiers.splice(index, 1)"
            >
              删除
            </el-button>
          </div>
          <el-button
            class="tier-add"
            @click="formData.tiers.push({ threshold: 0, faceValue: 0 })"
          >
            <el-icon><Plus /></el-icon>添加档位
          </el-button>
        </template>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 关联套餐弹窗 -->
    <el-dialog
      v-model="linkDialog.visible"
      title="关联套餐"
      width="520px"
      @close="closeLinkDialog"
    >
      <el-alert
        title="保存后将替换该活动的全部关联套餐；关联套餐参与活动价格计算。"
        type="info"
        :closable="false"
        class="link-tip"
      />
      <el-select
        v-model="linkDialog.packageIds"
        multiple
        filterable
        placeholder="选择要关联的套餐"
        style="width: 100%"
      >
        <el-option
          v-for="pkg in packageOptions"
          :key="pkg.id"
          :label="pkg.name"
          :value="pkg.id"
        />
      </el-select>

      <template #footer>
        <div class="dialog-footer">
          <el-button
            type="primary"
            :loading="linkDialog.loading"
            @click="handleLinkSubmit"
          >
            保 存
          </el-button>
          <el-button @click="closeLinkDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  PromotionAPI,
  PackageAPI,
  type PromotionQuery,
  type PromotionVO,
  type PackagePageVO,
} from "dehaze-sdk-js";
import type { TagType } from "@/enums/TagType";
import {
  Search,
  Refresh,
  Plus,
  Edit,
  Delete,
  Link,
} from "@element-plus/icons-vue";

defineOptions({
  name: "PackagePromotion",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const promotionFormRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);
const dateRange = ref<[string, string] | null>(null);

const queryParams = reactive<PromotionQuery>({
  pageNum: 1,
  pageSize: 10,
});

const promotionList = ref<PromotionVO[]>([]);
const packageOptions = ref<PackagePageVO[]>([]);

const dialog = reactive({
  title: "",
  visible: false,
});

interface Tier {
  threshold: number;
  faceValue: number;
}

const defaultFormData = {
  id: undefined as number | undefined,
  name: "",
  type: "discount" as PromotionVO["type"],
  startTime: "",
  endTime: "",
  description: "",
  newUserOnly: 0,
  discountType: "percent" as "percent" | "fixed" | "full_reduction",
  discountPercent: 10,
  discountFixedYuan: 0,
  tiers: [] as Tier[],
};

const formData = reactive({ ...defaultFormData, tiers: [] as Tier[] });

const rules = reactive({
  name: [{ required: true, message: "请输入活动名称", trigger: "blur" }],
  type: [{ required: true, message: "请选择活动类型", trigger: "change" }],
  startTime: [{ required: true, message: "请选择开始时间", trigger: "change" }],
  endTime: [
    { required: true, message: "请选择结束时间", trigger: "change" },
    {
      validator: (_rule: any, value: string, callback: any) => {
        if (value && formData.startTime && value <= formData.startTime) {
          callback(new Error("结束时间必须晚于开始时间"));
        } else {
          callback();
        }
      },
      trigger: "change",
    },
  ],
  discountType: [
    { required: true, message: "请选择折扣方式", trigger: "change" },
  ],
});

const linkDialog = reactive({
  visible: false,
  loading: false,
  promotionId: 0,
  packageIds: [] as number[],
});

const typeOptions: { label: string; value: string; tag: TagType }[] = [
  { label: "限时折扣", value: "discount", tag: "primary" },
  { label: "新用户专享", value: "new_user", tag: "warning" },
  { label: "节日促销", value: "holiday", tag: "danger" },
  { label: "满减活动", value: "full_reduction", tag: "success" },
];

function typeLabel(type: string) {
  return typeOptions.find((o) => o.value === type)?.label ?? type;
}

function typeTag(type: string): TagType {
  return typeOptions.find((o) => o.value === type)?.tag ?? "info";
}

function handleDateChange(value: [string, string] | null) {
  if (value) {
    queryParams.startTime = value[0];
    queryParams.endTime = value[1];
  } else {
    queryParams.startTime = undefined;
    queryParams.endTime = undefined;
  }
}

function handleQuery() {
  loading.value = true;
  PromotionAPI.getPage(queryParams)
    .then((data) => {
      promotionList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value.resetFields();
  dateRange.value = null;
  queryParams.startTime = undefined;
  queryParams.endTime = undefined;
  queryParams.pageNum = 1;
  handleQuery();
}

function isPromotionActive(row: PromotionVO) {
  const now = Date.now();
  return (
    row.status === 1 &&
    !!row.startTime &&
    !!row.endTime &&
    new Date(row.startTime.replace(/-/g, "/")).getTime() <= now &&
    new Date(row.endTime.replace(/-/g, "/")).getTime() >= now
  );
}

function buildRulesPayload() {
  if (formData.discountType === "percent") {
    return {
      discount_type: "percent",
      discount_value: formData.discountPercent,
    };
  }
  if (formData.discountType === "fixed") {
    return {
      discount_type: "fixed",
      discount_value: Math.round(formData.discountFixedYuan * 100),
    };
  }
  return {
    discount_type: "full_reduction",
    discount_value: 0,
    tiers: formData.tiers.map((t) => ({
      threshold: Math.round(t.threshold * 100),
      faceValue: Math.round(t.faceValue * 100),
    })),
  };
}

function openDialog(id?: number) {
  dialog.visible = true;
  if (id) {
    dialog.title = "编辑活动";
    const row = promotionList.value.find((p) => p.id === id);
    if (row) {
      const rulesData = row.activityRules ?? {};
      formData.id = row.id;
      formData.name = row.name;
      formData.type = row.type;
      formData.startTime = row.startTime;
      formData.endTime = row.endTime;
      formData.description = row.description ?? "";
      formData.newUserOnly = row.newUserOnly;
      const discountType = rulesData.discount_type ?? "percent";
      formData.discountType = discountType;
      formData.discountPercent = Number(rulesData.discount_value ?? 0) || 10;
      formData.discountFixedYuan =
        (Number(rulesData.discount_value ?? 0) || 0) / 100;
      formData.tiers = (rulesData.tiers ?? []).map((t: any) => ({
        threshold: (Number(t.threshold) || 0) / 100,
        faceValue: (Number(t.faceValue) || 0) / 100,
      }));
    }
  } else {
    dialog.title = "新增活动";
    Object.assign(formData, defaultFormData);
    formData.tiers = [];
    formData.id = undefined;
  }
}

function handleEdit(row: PromotionVO) {
  openDialog(row.id);
}

function closeDialog() {
  dialog.visible = false;
  promotionFormRef.value?.resetFields();
  promotionFormRef.value?.clearValidate();
}

function handleSubmit() {
  promotionFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    const row = promotionList.value.find((p) => p.id === formData.id);
    const submit = () => {
      const payload = {
        id: formData.id,
        name: formData.name,
        type: formData.type,
        description: formData.description,
        startTime: formData.startTime,
        endTime: formData.endTime,
        activityRules: buildRulesPayload(),
        newUserOnly: formData.newUserOnly,
      };
      const id = formData.id;
      const action = id
        ? PromotionAPI.update(id, payload)
        : PromotionAPI.add(payload);
      loading.value = true;
      action
        .then(() => {
          ElMessage.success(id ? "修改成功" : "新增成功");
          closeDialog();
          handleQuery();
        })
        .finally(() => {
          loading.value = false;
        });
    };
    // 进行中的活动修改规则会影响价格计算，需二次确认
    if (row && isPromotionActive(row)) {
      ElMessageBox.confirm(
        "该活动正在进行中，修改规则将影响在售套餐的价格计算，确认保存？",
        "二次确认",
        { confirmButtonText: "确定", cancelButtonText: "取消", type: "warning" }
      )
        .then(submit)
        .catch(() => {});
    } else {
      submit();
    }
  });
}

function handleToggleStatus(row: PromotionVO) {
  const next = row.status === 1 ? 0 : 1;
  const text = next === 1 ? "启用" : "禁用";
  ElMessageBox.confirm(`确认${text}活动「${row.name}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      loading.value = true;
      return PromotionAPI.updateStatus(row.id, next as 0 | 1);
    })
    .then(() => {
      ElMessage.success(`${text}成功`);
      handleQuery();
    })
    .catch(() => {})
    .finally(() => {
      loading.value = false;
    });
}

function handleDelete(row: PromotionVO) {
  ElMessageBox.confirm(
    `确认删除活动「${row.name}」吗？删除后不可恢复。`,
    "警告",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(() => {
      loading.value = true;
      return PromotionAPI.delete(row.id);
    })
    .then(() => {
      ElMessage.success("删除成功");
      handleQuery();
    })
    .catch(() => {})
    .finally(() => {
      loading.value = false;
    });
}

function loadPackageOptions() {
  if (packageOptions.value.length > 0) return;
  PackageAPI.getPage({ pageNum: 1, pageSize: 100 })
    .then((data) => {
      packageOptions.value = data.list;
    })
    .catch(() => {
      packageOptions.value = [];
    });
}

function openLinkDialog(row: PromotionVO) {
  loadPackageOptions();
  linkDialog.promotionId = row.id;
  linkDialog.packageIds = [];
  linkDialog.visible = true;
}

function closeLinkDialog() {
  linkDialog.visible = false;
  linkDialog.packageIds = [];
}

function handleLinkSubmit() {
  if (linkDialog.packageIds.length === 0) {
    ElMessage.warning("请选择要关联的套餐");
    return;
  }
  linkDialog.loading = true;
  PromotionAPI.bindPackages(linkDialog.promotionId, {
    packageIds: linkDialog.packageIds,
  })
    .then(() => {
      ElMessage.success("关联成功");
      closeLinkDialog();
    })
    .finally(() => {
      linkDialog.loading = false;
    });
}

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.form-tip {
  font-size: 12px;
  line-height: 1.4;
  color: var(--el-text-color-secondary);
}

.tier-row {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 4px 0 4px 120px;

  .tier-sep {
    color: var(--el-text-color-regular);
  }
}

.tier-add {
  margin-top: 8px;
  margin-left: 120px;
}

.link-tip {
  margin-bottom: 12px;
}
</style>
