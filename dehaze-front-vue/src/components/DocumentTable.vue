<!-- 文档列表 -->
<script lang="ts" setup>
import type { DocumentVO } from "dehaze-sdk-js";
import { ref } from "vue";

defineOptions({ name: "DocumentTable" });

defineProps<{
  documents: DocumentVO[];
  total: number;
  loading: boolean;
  readonly?: boolean;
}>();

const emit = defineEmits<{
  (e: "view", doc: DocumentVO): void;
  (e: "delete", doc: DocumentVO): void;
  (e: "reprocess", doc: DocumentVO): void;
  (e: "page-change", query: { pageNum: number; pageSize: number }): void;
}>();

// 分页状态在组件内部维护，经 page-change 通知宿主刷新
const pageNum = ref(1);
const pageSize = ref(10);

const SOURCE_LABELS: Record<string, string> = {
  upload: "文件上传",
  url: "网页导入",
  manual: "自定义文本",
  algorithm_sync: "算法同步",
  experience: "经验沉淀",
};

function handleSizeChange(size: number) {
  pageSize.value = size;
  pageNum.value = 1;
  emit("page-change", { pageNum: pageNum.value, pageSize: pageSize.value });
}

function handleCurrentChange(page: number) {
  pageNum.value = page;
  emit("page-change", { pageNum: pageNum.value, pageSize: pageSize.value });
}
</script>

<template>
  <div>
    <el-table v-loading="loading" :data="documents" border row-key="id">
      <el-table-column
        label="标题"
        prop="title"
        min-width="200"
        show-overflow-tooltip
      />
      <el-table-column label="状态" width="100" align="center">
        <template #default="{ row }">
          <DocumentStatusTags
            :status="(row as DocumentVO).processingStatus"
            :error="(row as DocumentVO).error"
          />
        </template>
      </el-table-column>
      <el-table-column label="来源" width="110" align="center">
        <template #default="{ row }">
          <span>{{
            SOURCE_LABELS[(row as DocumentVO).source] ??
            (row as DocumentVO).source
          }}</span>
        </template>
      </el-table-column>
      <el-table-column
        label="分块数"
        prop="chunkCount"
        width="80"
        align="center"
      />
      <el-table-column
        label="Token"
        prop="totalTokens"
        width="90"
        align="center"
      />
      <el-table-column
        label="更新时间"
        prop="updateTime"
        width="170"
        align="center"
      >
        <template #default="{ row }">
          <span>{{
            (row as DocumentVO).updateTime ?? (row as DocumentVO).createTime
          }}</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            type="primary"
            link
            size="small"
            @click="emit('view', row as DocumentVO)"
          >
            查看原文
          </el-button>
          <el-button
            v-if="(row as DocumentVO).processingStatus === 'failed'"
            type="warning"
            link
            size="small"
            @click="emit('reprocess', row as DocumentVO)"
          >
            重新处理
          </el-button>
          <el-button
            v-if="!readonly"
            type="danger"
            link
            size="small"
            @click="emit('delete', row as DocumentVO)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="flex justify-end mt-4">
      <el-pagination
        v-model:current-page="pageNum"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        background
        layout="total, sizes, prev, pager, next"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </div>
  </div>
</template>
