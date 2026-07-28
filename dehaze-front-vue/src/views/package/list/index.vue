<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="套餐名" prop="name">
          <el-input
            v-model="queryParams.name"
            clearable
            placeholder="套餐名"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="等级" prop="levelCode">
          <el-select
            v-model="queryParams.levelCode"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option label="基础版" value="level_1" />
            <el-option label="专业版" value="level_2" />
            <el-option label="旗舰版" value="level_3" />
          </el-select>
        </el-form-item>
        <el-form-item label="计费周期" prop="period">
          <el-select
            v-model="queryParams.period"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option label="月卡" value="monthly" />
            <el-option label="季卡" value="quarterly" />
            <el-option label="年卡" value="yearly" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-select
            v-model="queryParams.status"
            clearable
            placeholder="全部"
            style="width: 120px"
          >
            <el-option label="在售" :value="1" />
            <el-option label="下架" :value="0" />
          </el-select>
        </el-form-item>
        <el-form-item label="创建时间">
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
        <div class="flex justify-between items-center">
          <div>
            <el-button
              v-hasPerm="['package:add']"
              type="success"
              @click="openDialog()"
            >
              <el-icon><Plus /></el-icon>新增套餐
            </el-button>
            <el-button
              v-hasPerm="['package:delete']"
              :disabled="ids.length === 0"
              type="danger"
              @click="handleDelete()"
            >
              <el-icon><Delete /></el-icon>批量删除
            </el-button>
          </div>
          <el-button
            v-hasPerm="['package:sales']"
            type="warning"
            plain
            @click="openStatsDrawer"
          >
            <el-icon><TrendCharts /></el-icon>销售统计
          </el-button>
        </div>
      </template>

      <el-table
        ref="dataTableRef"
        v-loading="loading"
        :data="packageList"
        border
        highlight-current-row
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" align="center" />
        <el-table-column label="套餐名" prop="name" min-width="140" />
        <el-table-column label="等级" align="center" width="110">
          <template #default="scope">
            <el-tag
              :color="levelTagColor((scope.row as PackagePageVO).levelCode)"
              effect="dark"
              style="border: none"
            >
              {{ (scope.row as PackagePageVO).levelName }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="计费周期" align="center" width="100">
          <template #default="scope">
            {{ periodLabel((scope.row as PackagePageVO).period) }}
          </template>
        </el-table-column>
        <el-table-column label="原价" align="right" width="100">
          <template #default="scope">
            ¥{{ (scope.row as PackagePageVO).originalPrice.toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column label="售价" align="right" width="100">
          <template #default="scope">
            <span class="sale-price">
              ¥{{ (scope.row as PackagePageVO).salePrice.toFixed(2) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="日均" align="right" width="110">
          <template #default="scope">
            ¥{{ (scope.row as PackagePageVO).dailyPrice.toFixed(2) }}/天
          </template>
        </el-table-column>
        <el-table-column
          label="销量"
          prop="salesCount"
          align="center"
          width="90"
        />
        <el-table-column label="状态" align="center" width="90">
          <template #default="scope">
            <el-tag
              :type="
                (scope.row as PackagePageVO).status === 1 ? 'success' : 'info'
              "
              effect="plain"
            >
              {{ (scope.row as PackagePageVO).status === 1 ? "在售" : "下架" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="创建时间"
          prop="createTime"
          align="center"
          width="180"
        />
        <el-table-column label="操作" fixed="right" width="220" align="center">
          <template #default="scope">
            <el-button
              v-hasPerm="['package:edit']"
              link
              size="small"
              type="primary"
              @click="handleEdit(scope.row as PackagePageVO)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-hasPerm="['package:edit']"
              link
              size="small"
              :type="
                (scope.row as PackagePageVO).status === 1
                  ? 'warning'
                  : 'success'
              "
              @click="handleToggleStatus(scope.row as PackagePageVO)"
            >
              {{ (scope.row as PackagePageVO).status === 1 ? "下架" : "上架" }}
            </el-button>
            <el-button
              v-hasPerm="['package:delete']"
              link
              size="small"
              type="danger"
              @click="handleDelete(scope.row as PackagePageVO)"
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

    <!-- 套餐表单弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="720px"
      @close="closeDialog"
    >
      <el-form
        ref="packageFormRef"
        :model="formData"
        :rules="rules"
        label-width="120px"
      >
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="套餐名" prop="name">
              <el-input v-model="formData.name" placeholder="请输入套餐名" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="等级" prop="levelCode">
              <el-select v-model="formData.levelCode" style="width: 100%">
                <el-option label="基础版" value="level_1" />
                <el-option label="专业版" value="level_2" />
                <el-option label="旗舰版" value="level_3" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="计费周期" prop="period">
              <el-select
                v-model="formData.period"
                style="width: 100%"
                @change="handlePeriodChange"
              >
                <el-option label="月卡" value="monthly" />
                <el-option label="季卡" value="quarterly" />
                <el-option label="年卡" value="yearly" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="周期天数" prop="periodDays">
              <el-input-number
                v-model="formData.periodDays"
                :min="1"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="原价" prop="originalPrice">
              <el-input-number
                v-model="formData.originalPrice"
                :min="0"
                :precision="2"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="售价" prop="salePrice">
              <el-input-number
                v-model="formData.salePrice"
                :min="0"
                :precision="2"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="排序号" prop="sort">
              <el-input-number
                v-model="formData.sort"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="状态" prop="status">
              <el-switch
                v-model="formData.status"
                :active-value="1"
                :inactive-value="0"
                active-text="在售"
                inactive-text="下架"
                inline-prompt
              />
            </el-form-item>
          </el-col>
          <el-col :span="24">
            <el-form-item label="描述" prop="description">
              <el-input
                v-model="formData.description"
                type="textarea"
                :rows="2"
                placeholder="套餐描述"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">权益覆盖配置</el-divider>
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="去雾配额">
              <el-input-number
                v-model="benefitForm.monthlyDehazeQuota"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="评估配额">
              <el-input-number
                v-model="benefitForm.monthlyEvaluateQuota"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="历史保留(天)">
              <el-input-number
                v-model="benefitForm.historyRetention"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="批量上限">
              <el-input-number
                v-model="benefitForm.batchLimit"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="优先级">
              <el-input-number
                v-model="benefitForm.priority"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="高级参数">
              <el-input-number
                v-model="benefitForm.advancedParams"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="高清导出">
              <el-input-number
                v-model="benefitForm.hdExport"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="报告导出">
              <el-input-number
                v-model="benefitForm.reportExport"
                :min="0"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="批量下载">
              <el-input-number
                v-model="benefitForm.batchDownload"
                :min="0"
                controls-position="right"
                style="width: 100%"
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

    <!-- 销售统计抽屉 -->
    <el-drawer
      v-model="statsDrawer.visible"
      title="销售统计"
      direction="rtl"
      size="640px"
      @close="disposeCharts"
    >
      <div v-loading="statsDrawer.loading">
        <el-row :gutter="16" class="stats-overview">
          <el-col :span="12">
            <el-card shadow="hover" class="stats-card">
              <div class="stats-label">总销售额</div>
              <div class="stats-value">
                ¥{{ (statsData?.totalRevenue ?? 0).toFixed(2) }}
              </div>
            </el-card>
          </el-col>
          <el-col :span="12">
            <el-card shadow="hover" class="stats-card">
              <div class="stats-label">总销量</div>
              <div class="stats-value">{{ statsData?.totalSales ?? 0 }}</div>
            </el-card>
          </el-col>
        </el-row>

        <h4 class="stats-title">各套餐销售统计</h4>
        <div id="packageStatsChart" style="width: 100%; height: 240px"></div>

        <h4 class="stats-title">各等级统计</h4>
        <div id="levelStatsChart" style="width: 100%; height: 240px"></div>

        <h4 class="stats-title">各周期统计</h4>
        <div id="periodStatsChart" style="width: 100%; height: 240px"></div>

        <h4 class="stats-title">优惠券使用统计</h4>
        <el-card shadow="never" class="coupon-stats-card">
          <div class="coupon-stats-row">
            <span class="coupon-stats-label">累计发放：</span>
            <span class="coupon-stats-value">
              {{ statsData?.couponStats.totalIssued ?? 0 }}
            </span>
          </div>
          <div class="coupon-stats-row">
            <span class="coupon-stats-label">累计使用：</span>
            <span class="coupon-stats-value">
              {{ statsData?.couponStats.totalUsed ?? 0 }}
            </span>
          </div>
          <div class="coupon-stats-row">
            <span class="coupon-stats-label">使用率：</span>
            <span class="coupon-stats-value">
              {{ ((statsData?.couponStats.usageRate ?? 0) * 100).toFixed(1) }}%
            </span>
          </div>
        </el-card>
        <div id="couponUsageChart" style="width: 100%; height: 240px"></div>
      </div>
    </el-drawer>
  </div>
</template>

<script lang="ts" setup>
import {
  PackageAPI,
  type PackageQuery,
  type PackagePageVO,
  type PackageForm,
  type SalesStatsVO,
  type BenefitOverrides,
} from "dehaze-sdk-js";
import {
  Search,
  Refresh,
  Plus,
  Edit,
  Delete,
  TrendCharts,
} from "@element-plus/icons-vue";
import * as echarts from "echarts";

defineOptions({
  name: "PackageList",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const packageFormRef = ref(ElForm);

const loading = ref(false);
const ids = ref<number[]>([]);
const total = ref(0);
const dateRange = ref<[string, string] | null>(null);

const queryParams = reactive<PackageQuery>({
  pageNum: 1,
  pageSize: 10,
});

const packageList = ref<PackagePageVO[]>([]);

const dialog = reactive({
  title: "",
  visible: false,
});

const defaultFormData: PackageForm = {
  name: "",
  levelCode: "level_1",
  period: "monthly",
  periodDays: 30,
  originalPrice: 0,
  salePrice: 0,
  description: "",
  sort: 1,
  status: 1,
};

const formData = reactive<PackageForm>({ ...defaultFormData });

const defaultBenefitForm: BenefitOverrides = {
  monthlyDehazeQuota: 0,
  monthlyEvaluateQuota: 0,
  historyRetention: 0,
  batchLimit: 0,
  priority: 0,
  advancedParams: 0,
  hdExport: 0,
  reportExport: 0,
  batchDownload: 0,
};

const benefitForm = reactive<BenefitOverrides>({ ...defaultBenefitForm });

const statsDrawer = reactive({
  visible: false,
  loading: false,
});

const statsData = ref<SalesStatsVO>();

const packageStatsChart = ref<any>(null);
const levelStatsChart = ref<any>(null);
const periodStatsChart = ref<any>(null);
const couponUsageChart = ref<any>(null);

const rules = reactive({
  name: [{ required: true, message: "请输入套餐名", trigger: "blur" }],
  levelCode: [{ required: true, message: "请选择等级", trigger: "change" }],
  period: [{ required: true, message: "请选择计费周期", trigger: "change" }],
  originalPrice: [{ required: true, message: "请输入原价", trigger: "blur" }],
  salePrice: [{ required: true, message: "请输入售价", trigger: "blur" }],
});

const periodDaysMap: Record<string, number> = {
  monthly: 30,
  quarterly: 90,
  yearly: 365,
};

const levelTagColorMap: Record<string, string> = {
  level_1: "#409eff",
  level_2: "#722ed1",
  level_3: "#fa8c16",
};

function levelTagColor(levelCode: string) {
  return levelTagColorMap[levelCode] ?? "#409eff";
}

function periodLabel(period: string) {
  const map: Record<string, string> = {
    monthly: "月卡",
    quarterly: "季卡",
    yearly: "年卡",
  };
  return map[period] ?? period;
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
  PackageAPI.getPage(queryParams)
    .then((data) => {
      packageList.value = data.list;
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

function handleSelectionChange(selection: PackagePageVO[]) {
  ids.value = selection.map((item) => item.id);
}

function handlePeriodChange(value: string) {
  formData.periodDays = periodDaysMap[value] ?? 30;
}

function openDialog(id?: number) {
  dialog.visible = true;
  resetBenefitForm();
  if (id) {
    dialog.title = "编辑套餐";
    loading.value = true;
    PackageAPI.getForm(id)
      .then((data) => {
        Object.assign(formData, data);
        if (data.benefitOverrides) {
          Object.assign(benefitForm, data.benefitOverrides);
        }
      })
      .finally(() => {
        loading.value = false;
      });
  } else {
    dialog.title = "新增套餐";
    Object.assign(formData, defaultFormData);
    formData.id = undefined;
  }
}

function handleEdit(row: PackagePageVO) {
  openDialog(row.id);
}

function resetBenefitForm() {
  Object.assign(benefitForm, defaultBenefitForm);
}

function handleSubmit() {
  packageFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    loading.value = true;
    const submitData: PackageForm = {
      ...formData,
      benefitOverrides: { ...benefitForm },
    };
    const id = formData.id;
    const action = id
      ? PackageAPI.update(id, submitData)
      : PackageAPI.add(submitData);
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
  packageFormRef.value?.resetFields();
  packageFormRef.value?.clearValidate();
  resetBenefitForm();
}

function handleToggleStatus(row: PackagePageVO) {
  const next = row.status === 1 ? 0 : 1;
  const text = next === 1 ? "上架" : "下架";
  ElMessageBox.confirm(`确认${text}套餐「${row.name}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      loading.value = true;
      return PackageAPI.updateStatus(row.id, next);
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

function handleDelete(row?: PackagePageVO) {
  const packageIds = row?.id ? String(row.id) : ids.value.join(",");
  if (!packageIds) {
    ElMessage.warning("请勾选删除项");
    return;
  }
  const confirmText = row
    ? `确认删除套餐「${row.name}」吗？删除后不可恢复。`
    : "确认删除选中的套餐吗？删除后不可恢复。";
  ElMessageBox.confirm(confirmText, "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      loading.value = true;
      return PackageAPI.deleteByIds(packageIds);
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

function openStatsDrawer() {
  statsDrawer.visible = true;
  statsDrawer.loading = true;
  PackageAPI.getSalesStats()
    .then((data) => {
      statsData.value = data;
      nextTick(() => {
        initCharts(data);
      });
    })
    .finally(() => {
      statsDrawer.loading = false;
    });
}

function initCharts(data: SalesStatsVO) {
  const pkgEl = document.getElementById("packageStatsChart");
  if (pkgEl) {
    packageStatsChart.value = markRaw(echarts.init(pkgEl));
    packageStatsChart.value.setOption({
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
      },
      legend: { data: ["销售额", "销量"] },
      grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
      xAxis: {
        type: "category",
        data: data.packageStats.map((p) => p.packageName),
        axisPointer: { type: "shadow" },
      },
      yAxis: [
        {
          type: "value",
          name: "销售额",
          axisLabel: { formatter: "¥{value}" },
        },
        {
          type: "value",
          name: "销量",
          axisLabel: { formatter: "{value}" },
        },
      ],
      series: [
        {
          name: "销售额",
          type: "bar",
          data: data.packageStats.map((p) => p.revenue),
          barWidth: 20,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#83bff6" },
              { offset: 1, color: "#188df0" },
            ]),
          },
        },
        {
          name: "销量",
          type: "line",
          yAxisIndex: 1,
          data: data.packageStats.map((p) => p.salesCount),
          itemStyle: { color: "#67C23A" },
        },
      ],
    });
  }

  const levelEl = document.getElementById("levelStatsChart");
  if (levelEl) {
    levelStatsChart.value = markRaw(echarts.init(levelEl));
    const levelColorMap: Record<string, string> = {
      level_1: "#409eff",
      level_2: "#722ed1",
      level_3: "#fa8c16",
    };
    levelStatsChart.value.setOption({
      tooltip: {
        trigger: "item",
        formatter: "{a} <br/>{b}: ¥{c} ({d}%)",
      },
      legend: { orient: "vertical", left: "left" },
      series: [
        {
          name: "等级销售额",
          type: "pie",
          radius: "50%",
          data: data.levelStats.map((l) => ({
            name: l.levelName,
            value: l.revenue,
            itemStyle: { color: levelColorMap[l.levelCode] },
          })),
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    });
  }

  const periodEl = document.getElementById("periodStatsChart");
  if (periodEl) {
    periodStatsChart.value = markRaw(echarts.init(periodEl));
    periodStatsChart.value.setOption({
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
      },
      grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
      xAxis: {
        type: "category",
        data: data.periodStats.map((p) => p.periodName),
        axisPointer: { type: "shadow" },
      },
      yAxis: { type: "value", name: "销量" },
      series: [
        {
          name: "销量",
          type: "bar",
          data: data.periodStats.map((p) => p.salesCount),
          barWidth: 30,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#83bff6" },
              { offset: 1, color: "#188df0" },
            ]),
          },
        },
      ],
    });
  }

  const couponEl = document.getElementById("couponUsageChart");
  if (couponEl) {
    couponUsageChart.value = markRaw(echarts.init(couponEl));
    const rate = Math.round((data.couponStats.usageRate ?? 0) * 100);
    couponUsageChart.value.setOption({
      series: [
        {
          name: "使用率",
          type: "gauge",
          startAngle: 90,
          endAngle: -270,
          radius: "85%",
          pointer: { show: false },
          progress: {
            show: true,
            overlap: false,
            roundCap: true,
            clip: false,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                { offset: 0, color: "#83bff6" },
                { offset: 1, color: "#188df0" },
              ]),
            },
          },
          axisLine: {
            lineStyle: {
              width: 20,
              color: [[1, "#e6ebf5"]],
            },
          },
          splitLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          data: [{ value: rate, name: "使用率" }],
          title: {
            show: true,
            offsetCenter: [0, "30%"],
            fontSize: 13,
            color: "#666",
          },
          detail: {
            valueAnimation: true,
            formatter: "{value}%",
            fontSize: 22,
            offsetCenter: [0, 0],
            color: "#188df0",
          },
        },
      ],
    });
  }
}

function disposeCharts() {
  [
    packageStatsChart,
    levelStatsChart,
    periodStatsChart,
    couponUsageChart,
  ].forEach((c) => {
    if (c.value) {
      c.value.dispose();
      c.value = null;
    }
  });
}

function handleResize() {
  [
    packageStatsChart,
    levelStatsChart,
    periodStatsChart,
    couponUsageChart,
  ].forEach((c) => {
    if (c.value) c.value.resize();
  });
}

onMounted(() => {
  handleQuery();
  window.addEventListener("resize", handleResize);
});

onActivated(() => {
  handleResize();
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
  disposeCharts();
});
</script>

<style lang="scss" scoped>
.sale-price {
  font-weight: 600;
  color: var(--el-color-danger);
}

.stats-overview {
  margin-bottom: 16px;

  .stats-card {
    text-align: center;
  }

  .stats-label {
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  .stats-value {
    margin-top: 8px;
    font-size: 22px;
    font-weight: 700;
    color: var(--el-color-primary);
  }
}

.stats-title {
  margin: 20px 0 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.coupon-stats-card {
  .coupon-stats-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 13px;
  }

  .coupon-stats-label {
    color: var(--el-text-color-secondary);
  }

  .coupon-stats-value {
    font-weight: 600;
    color: var(--el-text-color-primary);
  }
}
</style>
