<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="用户名" prop="username">
          <el-input
            v-model="queryParams.username"
            clearable
            placeholder="用户名（精确匹配）"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <el-icon><Search /></el-icon>查询
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <el-table
        v-loading="loading"
        :data="sessionList"
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
        <el-table-column label="设备类型" width="130" align="center">
          <template #default="scope">
            <el-tag size="small" effect="plain">{{
              scope.row.deviceType || "web"
            }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="登录 IP" prop="ip" width="150" align="center" />
        <el-table-column
          label="登录时间"
          prop="loginTime"
          width="180"
          align="center"
        />
        <el-table-column
          label="最后访问时间"
          prop="lastAccessTime"
          width="180"
          align="center"
        />
        <el-table-column fixed="right" label="操作" width="120" align="center">
          <template #default="scope">
            <el-button
              v-hasPerm="['sys:auth:session:kick']"
              link
              size="small"
              type="danger"
              @click="handleKick(scope.row as SessionInfo)"
            >
              <el-icon><SwitchButton /></el-icon>踢出
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script lang="ts" setup>
import { AuthAPI, SessionInfo } from "dehaze-sdk-js";
import { Search, SwitchButton } from "@element-plus/icons-vue";

defineOptions({ name: "SystemOnlineSession" });

const queryFormRef = ref(ElForm);
const loading = ref(false);
const sessionList = ref<SessionInfo[]>([]);
const queryParams = reactive({ username: "" });

function handleQuery() {
  if (!queryParams.username.trim()) {
    ElMessage.warning("请输入用户名");
    return;
  }
  loading.value = true;
  AuthAPI.getSessions(queryParams.username.trim())
    .then((data) => {
      sessionList.value = data;
    })
    .finally(() => {
      loading.value = false;
    });
}

function handleKick(row: SessionInfo) {
  ElMessageBox.confirm("确认踢出该会话吗？被踢出端将立即下线。", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  })
    .then(() => AuthAPI.kickSession(row.sessionId))
    .then(() => {
      ElMessage.success("会话已踢出");
      handleQuery();
    })
    .catch(() => {});
}
</script>
