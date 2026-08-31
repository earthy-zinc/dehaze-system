<!-- 会话审计表格：scope=admin 全量用户会话（用户/模型/消息数/Token 与积分消耗/状态/异常标注） -->
<script lang="ts" setup>
import { Refresh } from "@element-plus/icons-vue";
import { storeToRefs } from "pinia";
import type { ConversationVO } from "dehaze-sdk-js";
import { useAdminAuditStore } from "@/store/modules/adminAudit";

defineOptions({ name: "ConversationAuditTable" });

const adminAuditStore = useAdminAuditStore();
const { auditList, auditTotal, auditFilter, auditLoading } =
  storeToRefs(adminAuditStore);

const PAGE_SIZES = [10, 20, 50];

function statusLabel(status: ConversationVO["status"]) {
  return status === 2 ? "已归档" : "活跃";
}

function statusTagType(status: ConversationVO["status"]) {
  return status === 2 ? "info" : "success";
}

function formatTime(time?: string) {
  if (!time) return "-";
  return time.slice(0, 16).replace("T", " ");
}

function handleSizeChange() {
  adminAuditStore.applyAuditFilter({});
}

function handlePageChange() {
  adminAuditStore.fetchAuditList();
}
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <div class="mb-4 flex items-center justify-between">
      <span class="font-bold">会话审计</span>
      <el-button
        v-has-perm="['ai:conversation:audit']"
        :icon="Refresh"
        @click="adminAuditStore.fetchAuditList()"
      >
        刷新
      </el-button>
    </div>

    <el-table v-loading="auditLoading" :data="auditList" border>
      <el-table-column label="ID" prop="id" width="80" />
      <el-table-column label="会话标题" min-width="200" show-overflow-tooltip>
        <template #default="{ row }">
          <span>{{ row.title }}</span>
          <el-tag
            v-if="row.matchedMessageId"
            type="warning"
            size="small"
            class="ml-1"
          >
            命中消息
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="用户" min-width="120" show-overflow-tooltip>
        <template #default="{ row }">
          {{ row.userName ?? `用户 ${row.userId ?? "-"}` }}
        </template>
      </el-table-column>
      <el-table-column
        label="模型"
        prop="model"
        min-width="140"
        show-overflow-tooltip
      >
        <template #default="{ row }">{{ row.model ?? "-" }}</template>
      </el-table-column>
      <el-table-column
        label="消息数"
        prop="messageCount"
        width="80"
        align="center"
      />
      <el-table-column label="Token 消耗" width="110" align="right">
        <template #default="{ row }">
          {{ row.tokenConsumed ?? "-" }}
        </template>
      </el-table-column>
      <el-table-column label="积分消耗" width="100" align="right">
        <template #default="{ row }">
          {{ row.creditsConsumed ?? "-" }}
        </template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-tag size="small" :type="statusTagType(row.status)">
            {{ statusLabel(row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="异常标注" min-width="120">
        <template #default="{ row }">
          <!-- 异常文案由后端 anomalyLabel 下发，前端不做硬编码映射 -->
          <el-tag v-if="row.anomalyLabel" type="danger" size="small">
            {{ row.anomalyLabel }}
          </el-tag>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="最后活跃" width="160">
        <template #default="{ row }">
          {{ formatTime(row.lastMessageAt ?? row.createTime) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="100" fixed="right" align="center">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            @click="
              adminAuditStore.openConversationDetail(row as ConversationVO)
            "
          >
            查看详情
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="auditFilter.pageNum"
      v-model:page-size="auditFilter.pageSize"
      :total="auditTotal"
      :page-sizes="PAGE_SIZES"
      layout="total, sizes, prev, pager, next"
      class="mt-4 justify-end"
      @size-change="handleSizeChange"
      @current-change="handlePageChange"
    />
  </el-card>
</template>
