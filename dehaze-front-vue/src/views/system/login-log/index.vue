<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="用户名" prop="username">
          <el-input
            v-model="queryParams.username"
            clearable
            placeholder="用户名"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="IP" prop="ip">
          <el-input
            v-model="queryParams.ip"
            clearable
            placeholder="IP 地址"
            style="width: 150px"
          />
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-select
            v-model="queryParams.status"
            clearable
            placeholder="全部"
            style="width: 110px"
          >
            <el-option label="成功" :value="1" />
            <el-option label="失败" :value="0" />
          </el-select>
        </el-form-item>
        <el-form-item label="设备类型" prop="deviceType">
          <el-select
            v-model="queryParams.deviceType"
            clearable
            placeholder="全部"
            style="width: 130px"
          >
            <el-option
              v-for="d in deviceTypes"
              :key="d"
              :label="d"
              :value="d"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="登录时间">
          <el-date-picker
            v-model="dateRange"
            type="daterange"
            range-separator="-"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
            style="width: 260px"
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
          v-hasPerm="['sys:auth:log:list']"
          type="success"
          :loading="exporting"
          @click="handleExport"
        >
          <el-icon><Download /></el-icon>导出 Excel
        </el-button>
      </template>

      <el-table
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
      >
        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column
          label="用户名"
          prop="username"
          width="140"
          show-overflow-tooltip
        />
        <el-table-column label="IP 地址" prop="ip" width="140" align="center" />
        <el-table-column label="设备类型" width="120" align="center">
          <template #default="scope">
            <el-tag size="small" effect="plain">{{
              scope.row.deviceType || "web"
            }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="浏览器"
          prop="browser"
          width="140"
          align="center"
        />
        <el-table-column
          label="操作系统"
          prop="os"
          width="140"
          align="center"
        />
        <el-table-column label="状态" width="90" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.status === 1 ? 'success' : 'danger'"
              size="small"
              effect="plain"
            >
              {{ scope.row.status === 1 ? "成功" : "失败" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="提示信息"
          prop="message"
          min-width="160"
          show-overflow-tooltip
        />
        <el-table-column
          label="登录时间"
          prop="loginTime"
          width="170"
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
import { AuthAPI, LoginLogQuery, LoginLogVO } from "dehaze-sdk-js";
import { Download, Refresh, Search } from "@element-plus/icons-vue";

defineOptions({ name: "SystemLoginLog" });

const deviceTypes = ["web", "android", "flutter", "miniprogram"];

const queryFormRef = ref(ElForm);
const loading = ref(false);
const exporting = ref(false);
const total = ref(0);
const pageData = ref<LoginLogVO[]>([]);
const dateRange = ref<[string, string] | null>(null);
const queryParams = reactive<LoginLogQuery>({
  pageNum: 1,
  pageSize: 10,
});

function buildQuery(): LoginLogQuery {
  return {
    ...queryParams,
    startTime: dateRange.value?.[0],
    endTime: dateRange.value?.[1],
  };
}

function handleQuery() {
  loading.value = true;
  AuthAPI.getLoginLogs(buildQuery())
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
  dateRange.value = null;
  queryParams.pageNum = 1;
  handleQuery();
}

function handleExport() {
  exporting.value = true;
  AuthAPI.exportLoginLogs(buildQuery())
    .then((blob) => {
      const url = URL.createObjectURL(new Blob([blob]));
      const link = document.createElement("a");
      link.href = url;
      link.download = `登录日志_${new Date().toISOString().slice(0, 10)}.xlsx`;
      link.click();
      URL.revokeObjectURL(url);
    })
    .finally(() => {
      exporting.value = false;
    });
}

onMounted(() => {
  handleQuery();
});
</script>
