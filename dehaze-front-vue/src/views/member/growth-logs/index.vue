<template>
  <div class="app-container growth-logs">
    <div class="page-header">
      <div class="header-title">
        <el-button link @click="router.push('/member/center')">
          <el-icon><ArrowLeft /></el-icon>
        </el-button>
        <span class="title-text">成长值明细</span>
      </div>
    </div>

    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="变动类型" prop="changeType">
          <el-select
            v-model="queryParams.changeType"
            clearable
            placeholder="全部"
            style="width: 150px"
          >
            <el-option
              v-for="item in changeTypeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="时间范围" prop="timeRange">
          <el-date-picker
            v-model="timeRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始"
            end-placeholder="结束"
            value-format="YYYY-MM-DD"
            @change="handleTimeChange"
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
      <el-table v-loading="loading" :data="pageData" border>
        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column label="变动类型" width="130" align="center">
          <template #default="scope">
            <span :class="['type-tag', `tag-${scope.row.changeType}`]">
              {{ getChangeTypeLabel(scope.row.changeType) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="变动值" width="100" align="center">
          <template #default="scope">
            <span
              :class="scope.row.changeValue >= 0 ? 'value-up' : 'value-down'"
            >
              {{ scope.row.changeValue >= 0 ? "+" : ""
              }}{{ scope.row.changeValue }}
            </span>
          </template>
        </el-table-column>
        <el-table-column
          label="变动后余额"
          prop="balance"
          width="120"
          align="center"
        />
        <el-table-column
          label="原因"
          prop="reason"
          min-width="200"
          show-overflow-tooltip
        />
        <el-table-column
          label="关联ID"
          prop="relatedId"
          width="120"
          align="center"
        >
          <template #default="scope">
            {{ scope.row.relatedId || "-" }}
          </template>
        </el-table-column>
        <el-table-column
          label="操作人"
          prop="operatorId"
          width="100"
          align="center"
        >
          <template #default="scope">
            {{ scope.row.operatorId ? `用户${scope.row.operatorId}` : "系统" }}
          </template>
        </el-table-column>
        <el-table-column
          label="时间"
          prop="createTime"
          width="180"
          align="center"
        />
      </el-table>

      <pagination
        v-if="total > 0"
        v-model:limit="queryParams.pageSize"
        v-model:page="queryParams.pageNum"
        v-model:total="total"
        @pagination="handleQuery"
      />
    </el-card>
  </div>
</template>

<script lang="ts" setup>
import {
  MemberAPI,
  GrowthLogQuery,
  GrowthLogVO,
  GrowthChangeType,
} from "dehaze-sdk-js";
import { ArrowLeft, Refresh, Search } from "@element-plus/icons-vue";

defineOptions({ name: "MemberGrowthLogs" });

const router = useRouter();
const queryFormRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);
const pageData = ref<GrowthLogVO[]>([]);
const timeRange = ref<string[]>([]);
const queryParams = reactive<GrowthLogQuery>({
  pageNum: 1,
  pageSize: 20,
});

const changeTypeOptions: Array<{ value: GrowthChangeType; label: string }> = [
  { value: "process", label: "图像处理" },
  { value: "evaluate", label: "指标评估" },
  { value: "rating", label: "评价奖励" },
  { value: "sign_in", label: "每日签到" },
  { value: "sign_in_bonus", label: "签到奖励" },
  { value: "consume", label: "消费获得" },
  { value: "refund_deduct", label: "退款扣除" },
  { value: "admin_adjust", label: "后台调整" },
];

function getChangeTypeLabel(type: GrowthChangeType) {
  return changeTypeOptions.find((t) => t.value === type)?.label ?? type;
}

function handleTimeChange(val: string[] | null) {
  if (val && val.length === 2) {
    queryParams.startTime = val[0] + " 00:00:00";
    queryParams.endTime = val[1] + " 23:59:59";
  } else {
    queryParams.startTime = undefined;
    queryParams.endTime = undefined;
  }
}

function handleQuery() {
  loading.value = true;
  MemberAPI.getGrowthLogs(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value?.resetFields();
  timeRange.value = [];
  queryParams.startTime = undefined;
  queryParams.endTime = undefined;
  queryParams.pageNum = 1;
  handleQuery();
}

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.growth-logs {
  max-width: 1000px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 16px;

  .header-title {
    display: flex;
    gap: 8px;
    align-items: center;

    .title-text {
      font-size: 22px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }
  }
}

.type-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  border-radius: 4px;

  &.tag-dehaze {
    color: #409eff;
    background: #ecf5ff;
  }

  &.tag-evaluate {
    color: #13c2c2;
    background: #e6fffb;
  }

  &.tag-rating {
    color: #fa8c16;
    background: #fff7e6;
  }

  &.tag-sign_in,
  &.tag-sign_in_bonus {
    color: #52c41a;
    background: #f6ffed;
  }

  &.tag-consume {
    color: #722ed1;
    background: #f9f0ff;
  }

  &.tag-refund_deduct {
    color: #f5222d;
    background: #fff1f0;
  }

  &.tag-admin_adjust {
    color: #8c8c8c;
    background: #fafafa;
  }
}

.value-up {
  font-weight: 600;
  color: #52c41a;
}

.value-down {
  font-weight: 600;
  color: #f5222d;
}
</style>
