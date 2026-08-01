<template>
  <div class="app-container">
    <el-card shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <span class="title">我的收藏</span>
          <el-radio-group
            v-model="currentTab"
            size="small"
            @change="handleTabChange"
          >
            <el-radio-button label="all">全部</el-radio-button>
            <el-radio-button label="algorithm">算法</el-radio-button>
            <el-radio-button label="result">处理结果</el-radio-button>
            <el-radio-button label="dataset">数据集</el-radio-button>
            <el-radio-button label="image">图片</el-radio-button>
            <el-radio-button label="preset">预设</el-radio-button>
          </el-radio-group>
        </div>
      </template>

      <div class="filter-bar">
        <el-input
          v-model="queryParams.keywords"
          placeholder="搜索收藏名称"
          clearable
          style="width: 240px"
          @keyup.enter="handleQuery"
        />
        <el-select
          v-model="queryParams.sortBy"
          placeholder="排序"
          style="width: 160px"
          @change="handleQuery"
        >
          <el-option label="按时间" value="createTime" />
          <el-option label="按评分" value="rating" />
          <el-option label="按使用频率" value="usageCount" />
        </el-select>
      </div>

      <div v-loading="loading" class="favorite-grid">
        <div v-if="pageData.length === 0 && !loading" class="empty-state">
          <el-empty description="暂无收藏内容" />
        </div>

        <el-card
          v-for="item in pageData"
          :key="`${item.targetType}-${item.targetId}`"
          class="favorite-card"
          shadow="hover"
          @click="handleCardClick(item)"
        >
          <template #header>
            <div class="card-header">
              <el-tag
                :type="getTargetTypeBadgeType(item.targetType)"
                size="small"
              >
                {{ getTargetTypeLabel(item.targetType) }}
              </el-tag>
              <el-tag
                v-if="item.isInvalid"
                type="danger"
                size="small"
                effect="dark"
              >
                已失效
              </el-tag>
            </div>
          </template>

          <div class="card-body">
            <div v-if="item.targetThumbnail" class="thumbnail">
              <el-image
                :src="item.targetThumbnail"
                fit="cover"
                :preview-src-list="[item.targetThumbnail]"
                style="width: 100%; height: 100%"
              />
            </div>
            <div
              class="card-info"
              :class="{ 'has-thumb': item.targetThumbnail }"
            >
              <div class="card-title" :title="item.targetName || ''">
                {{ item.targetName || "-" }}
              </div>
              <div class="card-summary" :title="item.targetSummary || ''">
                {{ item.targetSummary || "无摘要" }}
              </div>
              <div class="card-meta">
                <span>{{ formatDate(item.createTime) }}</span>
              </div>
            </div>
          </div>

          <template #footer>
            <div class="card-footer">
              <el-button
                link
                type="danger"
                size="small"
                @click.stop="handleCancelFavorite(item.id, $event)"
              >
                <el-icon><Delete /></el-icon>取消收藏
              </el-button>
            </div>
          </template>
        </el-card>
      </div>

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
import { ref, reactive, watch, onMounted } from "vue";
import { useRouter } from "vue-router";
import {
  FavoriteAPI,
  FavoriteQuery,
  FavoriteVO,
  FavoriteTargetType,
} from "dehaze-sdk-js";
import { Delete } from "@element-plus/icons-vue";
import { usePagination } from "@/composables/usePagination";

defineOptions({ name: "MyFavorites" });

const router = useRouter();
const { pageNum, pageSize, total, handlePageChange, handleSizeChange, reset } =
  usePagination({
    initialPageSize: 12,
  });

const currentTab = ref("all");
const loading = ref(false);
const pageData = ref<FavoriteVO[]>([]);

const queryParams = reactive<FavoriteQuery>({
  pageNum: 1,
  pageSize: 12,
  sortBy: "createTime",
  sortOrder: "desc",
});

function getTargetTypeLabel(type: FavoriteTargetType): string {
  const labels: Record<FavoriteTargetType, string> = {
    algorithm: "算法",
    result: "处理结果",
    dataset: "数据集",
    image: "图片",
    preset: "预设",
  };
  return labels[type] || type;
}

function getTargetTypeBadgeType(
  type: FavoriteTargetType
): "info" | "primary" | "success" | "warning" | "danger" {
  const types: Record<
    FavoriteTargetType,
    "info" | "primary" | "success" | "warning" | "danger"
  > = {
    algorithm: "primary",
    result: "success",
    dataset: "warning",
    image: "danger",
    preset: "info",
  };
  return types[type] || "info";
}

function formatDate(dateStr: string): string {
  return dateStr.substring(0, 10);
}

function handleTabChange() {
  queryParams.targetType =
    currentTab.value === "all"
      ? undefined
      : (currentTab.value as FavoriteTargetType);
  queryParams.pageNum = 1;
  handleQuery();
}

function handleQuery() {
  loading.value = true;
  queryParams.pageNum = pageNum.value;
  queryParams.pageSize = pageSize.value;

  FavoriteAPI.getPage(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function handleCardClick(item: FavoriteVO) {
  if (item.isInvalid) return;
  const routeMap: Record<FavoriteTargetType, string> = {
    algorithm: "/algorithm/detail",
    result: "/compare/result",
    dataset: "/dataset/detail",
    image: "/image-detail",
    preset: "/preset/detail",
  };
  const route = routeMap[item.targetType];
  if (route) {
    router.push(`${route}/${item.targetId}`);
  }
}

function handleCancelFavorite(id: number, event: MouseEvent) {
  event.stopPropagation();
  ElMessageBox.confirm("确认取消该收藏吗？", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() =>
      FavoriteAPI.deleteByIds([id]).then(() => {
        ElMessage.success("已取消收藏");
        handleQuery();
      })
    )
    .catch(() => {});
}

watch([pageNum, pageSize], () => {
  handleQuery();
});

onMounted(() => {
  handleQuery();
});
</script>

<style scoped>
.title {
  font-size: 18px;
  font-weight: 600;
}

.filter-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.favorite-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.empty-state {
  grid-column: 1 / -1;
  padding: 60px 0;
  text-align: center;
}

.favorite-card {
  cursor: pointer;
  transition: transform 0.2s;
}

.favorite-card:hover {
  transform: translateY(-2px);
}

.card-header {
  display: flex;
  gap: 8px;
  align-items: center;
}

.card-body {
  display: flex;
  gap: 12px;
}

.thumbnail {
  flex-shrink: 0;
  width: 80px;
  height: 80px;
  overflow: hidden;
  border-radius: 6px;
}

.card-info {
  flex: 1;
  min-width: 0;
}

.card-info.has-thumb {
  /* ensures proper flex behavior */
}

.card-title {
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  white-space: nowrap;
}

.card-summary {
  margin-bottom: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 12px;
  color: #909399;
  white-space: nowrap;
}

.card-meta {
  font-size: 12px;
  color: #c0c4cc;
}

.card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
</style>
