<template>
  <div class="app-container my-ratings">
    <div class="page-header">
      <span class="title-text">我的评价</span>
    </div>

    <div v-loading="loading" class="rating-list">
      <template v-if="ratingList.length > 0">
        <div v-for="rating in ratingList" :key="rating.id" class="rating-card">
          <div class="card-header">
            <div class="header-left">
              <el-icon class="algo-icon"><Star /></el-icon>
              <span class="algo-name">{{ rating.algorithmName }}</span>
            </div>
            <div class="header-right">
              <el-rate
                :model-value="rating.rating"
                disabled
                show-score
                score-template="{value}"
              />
              <span class="time-text">{{ rating.createTime }}</span>
            </div>
          </div>

          <div v-if="rating.comment" class="card-comment">
            {{ rating.comment }}
          </div>

          <div v-if="rating.tags?.length" class="card-tags">
            <el-tag
              v-for="tag in rating.tags"
              :key="tag"
              size="small"
              effect="light"
            >
              {{ tag }}
            </el-tag>
          </div>

          <div v-if="rating.imageUrls?.length" class="card-images">
            <el-image
              v-for="(url, idx) in rating.imageUrls"
              :key="idx"
              :src="url"
              :preview-src-list="rating.imageUrls"
              :initial-index="idx"
              fit="cover"
              class="thumb-img"
            />
          </div>

          <el-card v-if="rating.adminReply" class="reply-card" shadow="never">
            <div class="reply-header">
              <el-icon><ChatLineRound /></el-icon>
              <span>管理员回复</span>
              <span v-if="rating.replyTime" class="reply-time">
                {{ rating.replyTime }}
              </span>
            </div>
            <div class="reply-content">{{ rating.adminReply }}</div>
          </el-card>
        </div>
      </template>

      <el-empty v-else-if="!loading" description="暂无评价" :image-size="120" />
    </div>

    <pagination
      v-if="total > 0"
      v-model:limit="queryParams.pageSize"
      v-model:page="queryParams.pageNum"
      v-model:total="total"
      @pagination="handleQuery"
    />
  </div>
</template>

<script lang="ts" setup>
import { FeedbackAPI, MyRatingVO } from "dehaze-sdk-js";
import { ChatLineRound, Star } from "@element-plus/icons-vue";

defineOptions({ name: "FeedbackMyRatings" });

const loading = ref(false);
const total = ref(0);
const ratingList = ref<MyRatingVO[]>([]);
const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
});

function handleQuery() {
  loading.value = true;
  FeedbackAPI.listMyRatings(queryParams)
    .then((data) => {
      ratingList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.my-ratings {
  max-width: 960px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 20px;

  .title-text {
    font-size: 22px;
    font-weight: 600;
    color: var(--el-text-color-primary);
    letter-spacing: 0.5px;
  }
}

.rating-list {
  min-height: 240px;
}

.rating-card {
  padding: 18px 20px;
  margin-bottom: 14px;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  transition: all 0.25s ease;

  &:hover {
    border-color: var(--el-color-primary-light-5);
    box-shadow: 0 4px 16px rgb(0 0 0 / 6%);
  }
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;

  .header-left {
    display: flex;
    gap: 8px;
    align-items: center;

    .algo-icon {
      font-size: 18px;
      color: var(--el-color-primary);
    }

    .algo-name {
      font-size: 16px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }
  }

  .header-right {
    display: flex;
    gap: 14px;
    align-items: center;

    .time-text {
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }
}

.card-comment {
  padding: 10px 12px;
  margin-bottom: 12px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--el-text-color-regular);
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  background: var(--el-fill-color-light);
  border-radius: 6px;
}

.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.card-images {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;

  .thumb-img {
    width: 100px;
    height: 100px;
    border-radius: 6px;
  }
}

.reply-card {
  background: var(--el-fill-color-light);

  :deep(.el-card__body) {
    padding: 12px 14px;
  }

  .reply-header {
    display: flex;
    gap: 6px;
    align-items: center;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--el-color-primary);

    .reply-time {
      margin-left: auto;
      font-size: 12px;
      font-weight: 400;
      color: var(--el-text-color-secondary);
    }
  }

  .reply-content {
    font-size: 13px;
    line-height: 1.6;
    color: var(--el-text-color-regular);
    overflow-wrap: anywhere;
    white-space: pre-wrap;
  }
}
</style>
