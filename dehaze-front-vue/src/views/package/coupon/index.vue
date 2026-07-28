<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="名称" prop="name">
          <el-input
            v-model="queryParams.name"
            clearable
            placeholder="优惠券名称"
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
            <el-option label="满减券" value="full_reduction" />
            <el-option label="折扣券" value="discount" />
            <el-option label="无门槛券" value="no_threshold" />
            <el-option label="体验券" value="trial" />
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
        <div class="flex justify-between items-center">
          <el-button
            v-hasPerm="['coupon:add']"
            type="success"
            @click="openDialog()"
          >
            <el-icon><Plus /></el-icon>新增优惠券
          </el-button>
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="couponList"
        border
        highlight-current-row
      >
        <el-table-column label="名称" prop="name" min-width="160" />
        <el-table-column label="类型" align="center" width="100">
          <template #default="scope">
            <el-tag
              :type="couponTypeTag((scope.row as CouponVO).type)"
              effect="plain"
            >
              {{ couponTypeLabel((scope.row as CouponVO).type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="面值" align="right" width="100">
          <template #default="scope">
            <span class="face-value">
              {{ formatFaceValue(scope.row as CouponVO) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="门槛" align="right" width="100">
          <template #default="scope">
            <span v-if="(scope.row as CouponVO).threshold">
              满¥{{ ((scope.row as CouponVO).threshold ?? 0).toFixed(2) }}
            </span>
            <span v-else class="text-secondary">无门槛</span>
          </template>
        </el-table-column>
        <el-table-column label="有效期" align="center" min-width="200">
          <template #default="scope">
            <span v-if="(scope.row as CouponVO).validType === 'fixed'">
              {{ (scope.row as CouponVO).validStart }} ~
              {{ (scope.row as CouponVO).validEnd }}
            </span>
            <span v-else>
              领取后 {{ (scope.row as CouponVO).validDays ?? 0 }} 天
            </span>
          </template>
        </el-table-column>
        <el-table-column label="总量/已领/已用" align="center" width="140">
          <template #default="scope">
            {{ (scope.row as CouponVO).totalQty }} /
            {{ (scope.row as CouponVO).issuedQty }} /
            {{ (scope.row as CouponVO).usedQty }}
          </template>
        </el-table-column>
        <el-table-column
          label="每人限领"
          prop="perUserLimit"
          align="center"
          width="90"
        />
        <el-table-column label="状态" align="center" width="90">
          <template #default="scope">
            <el-tag
              :type="(scope.row as CouponVO).status === 1 ? 'success' : 'info'"
              effect="plain"
            >
              {{ (scope.row as CouponVO).status === 1 ? "启用" : "禁用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" fixed="right" width="220" align="center">
          <template #default="scope">
            <el-button
              v-hasPerm="['coupon:edit']"
              link
              size="small"
              type="primary"
              @click="handleEdit(scope.row as CouponVO)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-hasPerm="['coupon:distribute']"
              link
              size="small"
              type="warning"
              @click="openDistributeDialog(scope.row as CouponVO)"
            >
              <el-icon><Ticket /></el-icon>发放
            </el-button>
            <el-button
              v-hasPerm="['coupon:delete']"
              link
              size="small"
              type="danger"
              @click="handleDelete(scope.row as CouponVO)"
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

    <!-- 优惠券表单弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="680px"
      @close="closeDialog"
    >
      <el-form
        ref="couponFormRef"
        :model="formData"
        :rules="rules"
        label-width="120px"
      >
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="名称" prop="name">
              <el-input v-model="formData.name" placeholder="优惠券名称" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="类型" prop="type">
              <el-select v-model="formData.type" style="width: 100%">
                <el-option label="满减券" value="full_reduction" />
                <el-option label="折扣券" value="discount" />
                <el-option label="无门槛券" value="no_threshold" />
                <el-option label="体验券" value="trial" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="面值" prop="faceValue">
              <el-input-number
                v-model="formData.faceValue"
                :min="0"
                :precision="2"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="门槛" prop="threshold">
              <el-input-number
                v-model="formData.threshold"
                :min="0"
                :precision="2"
                :disabled="formData.type !== 'full_reduction'"
                controls-position="right"
                style="width: 100%"
                placeholder="满减必填"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="有效期类型" prop="validType">
              <el-radio-group v-model="formData.validType">
                <el-radio value="fixed">固定日期</el-radio>
                <el-radio value="relative">相对天数</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-col>
          <template v-if="formData.validType === 'fixed'">
            <el-col :span="12">
              <el-form-item label="生效时间" prop="validStart">
                <el-date-picker
                  v-model="formData.validStart"
                  type="datetime"
                  value-format="YYYY-MM-DD HH:mm:ss"
                  placeholder="生效时间"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="失效时间" prop="validEnd">
                <el-date-picker
                  v-model="formData.validEnd"
                  type="datetime"
                  value-format="YYYY-MM-DD HH:mm:ss"
                  placeholder="失效时间"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </template>
          <el-col v-else :span="12">
            <el-form-item label="有效天数" prop="validDays">
              <el-input-number
                v-model="formData.validDays"
                :min="1"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="总量" prop="totalQty">
              <el-input-number
                v-model="formData.totalQty"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="每人限领" prop="perUserLimit">
              <el-input-number
                v-model="formData.perUserLimit"
                :min="1"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="24">
            <el-form-item label="适用套餐" prop="applicableScope">
              <el-select
                v-model="formData.applicableScope"
                multiple
                clearable
                placeholder="不选则全部套餐适用"
                style="width: 100%"
              >
                <el-option
                  v-for="pkg in packageOptions"
                  :key="pkg.id"
                  :label="pkg.name"
                  :value="pkg.id"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="状态" prop="status">
              <el-switch
                v-model="formData.status"
                :active-value="1"
                :inactive-value="0"
                active-text="启用"
                inactive-text="禁用"
                inline-prompt
              />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 发放弹窗 -->
    <el-dialog
      v-model="distributeDialog.visible"
      title="发放优惠券"
      width="520px"
      @close="closeDistributeDialog"
    >
      <el-form
        ref="distributeFormRef"
        :model="distributeForm"
        :rules="distributeRules"
        label-width="120px"
      >
        <el-form-item label="优惠券">
          <el-input :model-value="distributeDialog.couponName" disabled />
        </el-form-item>
        <el-form-item label="发放范围" prop="targetScope">
          <el-radio-group v-model="distributeForm.targetScope">
            <el-radio value="all">全体用户</el-radio>
            <el-radio value="level">按等级</el-radio>
            <el-radio value="users">指定用户</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item
          v-if="distributeForm.targetScope === 'level'"
          label="会员等级"
          prop="levelCodes"
        >
          <el-select
            v-model="distributeForm.levelCodes"
            multiple
            placeholder="选择等级"
            style="width: 100%"
          >
            <el-option label="基础版" value="level_1" />
            <el-option label="专业版" value="level_2" />
            <el-option label="旗舰版" value="level_3" />
          </el-select>
        </el-form-item>
        <el-form-item
          v-if="distributeForm.targetScope === 'users'"
          label="用户ID"
          prop="userIdsInput"
        >
          <el-input
            v-model="userIdsInput"
            type="textarea"
            :rows="3"
            placeholder="多个用户ID用英文逗号分隔，如 1001,1002,1003"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button
            type="primary"
            :loading="distributeDialog.loading"
            @click="handleDistributeSubmit"
          >
            确认发放
          </el-button>
          <el-button @click="closeDistributeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  CouponAPI,
  PackageAPI,
  type CouponQuery,
  type CouponVO,
  type CouponForm,
  type CouponBatchDistributeForm,
  type PackagePageVO,
} from "dehaze-sdk-js";
import {
  Search,
  Refresh,
  Plus,
  Edit,
  Delete,
  Ticket,
} from "@element-plus/icons-vue";

defineOptions({
  name: "PackageCoupon",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const couponFormRef = ref(ElForm);
const distributeFormRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);

const queryParams = reactive<CouponQuery>({
  pageNum: 1,
  pageSize: 10,
});

const couponList = ref<CouponVO[]>([]);
const packageOptions = ref<PackagePageVO[]>([]);

const dialog = reactive({
  title: "",
  visible: false,
});

const defaultFormData: CouponForm = {
  name: "",
  type: "full_reduction",
  faceValue: 0,
  threshold: 0,
  validType: "fixed",
  validStart: undefined,
  validEnd: undefined,
  validDays: 7,
  totalQty: 100,
  perUserLimit: 1,
  applicableScope: [],
  status: 1,
};

const formData = reactive<CouponForm>({ ...defaultFormData });

const rules = reactive({
  name: [{ required: true, message: "请输入优惠券名称", trigger: "blur" }],
  type: [{ required: true, message: "请选择类型", trigger: "change" }],
  faceValue: [{ required: true, message: "请输入面值", trigger: "blur" }],
  threshold: [
    {
      validator: (_rule: any, value: number, callback: any) => {
        if (formData.type === "full_reduction" && (!value || value <= 0)) {
          callback(new Error("满减券必须填写门槛"));
        } else {
          callback();
        }
      },
      trigger: "blur",
    },
  ],
  validType: [
    { required: true, message: "请选择有效期类型", trigger: "change" },
  ],
  totalQty: [{ required: true, message: "请输入总量", trigger: "blur" }],
  perUserLimit: [
    { required: true, message: "请输入每人限领数", trigger: "blur" },
  ],
});

const distributeDialog = reactive({
  visible: false,
  loading: false,
  couponId: 0,
  couponName: "",
});

const defaultDistributeForm: CouponBatchDistributeForm = {
  couponId: 0,
  targetScope: "all",
  levelCodes: [],
  userIds: [],
};

const distributeForm = reactive<CouponBatchDistributeForm>({
  ...defaultDistributeForm,
});

const userIdsInput = ref("");

const distributeRules = reactive({
  targetScope: [
    { required: true, message: "请选择发放范围", trigger: "change" },
  ],
  levelCodes: [
    {
      validator: (_rule: any, value: string[], callback: any) => {
        if (
          distributeForm.targetScope === "level" &&
          (!value || value.length === 0)
        ) {
          callback(new Error("请选择会员等级"));
        } else {
          callback();
        }
      },
      trigger: "change",
    },
  ],
  userIdsInput: [
    {
      validator: (_rule: any, _value: string, callback: any) => {
        if (distributeForm.targetScope === "users") {
          const ids = parseUserIds(userIdsInput.value);
          if (ids.length === 0) {
            callback(new Error("请输入用户ID"));
          } else {
            callback();
          }
        } else {
          callback();
        }
      },
      trigger: "blur",
    },
  ],
});

type TagType = "primary" | "success" | "info" | "warning" | "danger";

const couponTypeLabelMap: Record<string, string> = {
  full_reduction: "满减券",
  discount: "折扣券",
  no_threshold: "无门槛券",
  trial: "体验券",
};

const couponTypeTagMap: Record<string, TagType> = {
  full_reduction: "warning",
  discount: "success",
  no_threshold: "primary",
  trial: "danger",
};

function couponTypeLabel(type: string) {
  return couponTypeLabelMap[type] ?? type;
}

function couponTypeTag(type: string): TagType {
  return couponTypeTagMap[type] ?? "info";
}

function formatFaceValue(coupon: CouponVO) {
  if (coupon.type === "discount") {
    return `${coupon.faceValue}折`;
  }
  return `¥${coupon.faceValue.toFixed(2)}`;
}

function parseUserIds(input: string): number[] {
  return input
    .split(/[,，\s]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number(s))
    .filter((n) => !Number.isNaN(n) && n > 0);
}

function handleQuery() {
  loading.value = true;
  CouponAPI.getPage(queryParams)
    .then((data) => {
      couponList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value.resetFields();
  queryParams.pageNum = 1;
  handleQuery();
}

function loadPackageOptions() {
  PackageAPI.getPage({ pageNum: 1, pageSize: 100 })
    .then((data) => {
      packageOptions.value = data.list;
    })
    .catch(() => {
      packageOptions.value = [];
    });
}

function openDialog(id?: number) {
  dialog.visible = true;
  if (packageOptions.value.length === 0) {
    loadPackageOptions();
  }
  if (id) {
    dialog.title = "编辑优惠券";
    const row = couponList.value.find((c) => c.id === id);
    if (row) {
      Object.assign(formData, {
        id: row.id,
        name: row.name,
        type: row.type,
        faceValue: row.faceValue,
        threshold: row.threshold,
        validType: row.validType,
        validStart: row.validStart,
        validEnd: row.validEnd,
        validDays: row.validDays,
        totalQty: row.totalQty,
        perUserLimit: row.perUserLimit,
        applicableScope: row.applicableScope ?? [],
        status: row.status,
      });
    }
  } else {
    dialog.title = "新增优惠券";
    Object.assign(formData, defaultFormData);
    formData.id = undefined;
  }
}

function handleEdit(row: CouponVO) {
  openDialog(row.id);
}

function handleSubmit() {
  couponFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    loading.value = true;
    const id = formData.id;
    const action = id
      ? CouponAPI.update(id, formData)
      : CouponAPI.add(formData);
    action
      .then(() => {
        ElMessage.success(id ? "修改成功" : "新增成功");
        closeDialog();
        resetQuery();
      })
      .finally(() => {
        loading.value = false;
      });
  });
}

function closeDialog() {
  dialog.visible = false;
  couponFormRef.value?.resetFields();
  couponFormRef.value?.clearValidate();
}

function handleDelete(row: CouponVO) {
  ElMessageBox.confirm(
    `确认删除优惠券「${row.name}」吗？删除后不可恢复。`,
    "警告",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(() => {
      loading.value = true;
      return CouponAPI.deleteByIds(String(row.id));
    })
    .then(() => {
      ElMessage.success("删除成功");
      resetQuery();
    })
    .catch(() => {})
    .finally(() => {
      loading.value = false;
    });
}

function openDistributeDialog(row: CouponVO) {
  distributeDialog.visible = true;
  distributeDialog.couponId = row.id;
  distributeDialog.couponName = row.name;
  Object.assign(distributeForm, defaultDistributeForm);
  userIdsInput.value = "";
}

function closeDistributeDialog() {
  distributeDialog.visible = false;
  distributeFormRef.value?.resetFields();
  distributeFormRef.value?.clearValidate();
  userIdsInput.value = "";
}

function handleDistributeSubmit() {
  distributeFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    distributeDialog.loading = true;
    const payload: CouponBatchDistributeForm = {
      couponId: distributeDialog.couponId,
      targetScope: distributeForm.targetScope,
    };
    if (distributeForm.targetScope === "level") {
      payload.levelCodes = distributeForm.levelCodes;
    } else if (distributeForm.targetScope === "users") {
      payload.userIds = parseUserIds(userIdsInput.value);
    }
    CouponAPI.batchDistribute(payload)
      .then((res) => {
        ElMessage.success(
          `发放完成：成功 ${res.successCount} 个，失败 ${res.failCount} 个`
        );
        closeDistributeDialog();
        handleQuery();
      })
      .finally(() => {
        distributeDialog.loading = false;
      });
  });
}

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.face-value {
  font-weight: 600;
  color: var(--el-color-danger);
}

.text-secondary {
  color: var(--el-text-color-secondary);
}
</style>
