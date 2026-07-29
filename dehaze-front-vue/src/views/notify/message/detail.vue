<template>
  <div class="app-container message-detail">
    <div class="detail-wrapper" v-loading="loading">
      <template v-if="message">
        <div class="detail-header">
          <el-button link @click="goBack">
            <el-icon><ArrowLeft /></el-icon>
            返回列表
          </el-button>
        </div>

        <div :class="['detail-card', `type-${message.type}`]">
          <div class="card-stripe"></div>
          <div class="card-content">
            <div class="meta-row">
              <span :class="['type-tag', `tag-${message.type}`]">
                <el-icon v-if="typeIconMap[message.type]">
                  <component :is="typeIconMap[message.type]" />
                </el-icon>
                {{ message.typeLabel }}
              </span>
              <span v-if="message.priority >= 3" class="priority-flag">
                <el-icon><WarnTriangleFilled /></el-icon>
                {{ message.priority === 4 ? "紧急" : "高优" }}
              </span>
              <span v-if="message.senderTypeLabel" class="sender-text">
                来自：{{ message.senderTypeLabel }}
              </span>
              <span class="time-text">{{ message.createTime }}</span>
            </div>

            <h1 class="detail-title">{{ message.title }}</h1>

            <div
              v-if="message.readStatus === 1 && message.readTime"
              class="read-info"
            >
              <el-icon><CircleCheckFilled /></el-icon>
              已读于 {{ message.readTime }}
            </div>

            <el-divider />

            <div class="detail-content">{{ message.content }}</div>

            <div v-if="message.extra" class="extra-block">
              <div class="extra-title">附加信息</div>
              <pre class="extra-content">{{
                JSON.stringify(message.extra, null, 2)
              }}</pre>
            </div>

            <div class="detail-footer">
              <el-button
                v-if="message.jumpUrl"
                type="primary"
                @click="handleJump"
              >
                <el-icon><Position /></el-icon>
                查看详情
              </el-button>
              <el-button type="danger" plain @click="handleDelete">
                <el-icon><Delete /></el-icon>
                删除消息
              </el-button>
            </div>
          </div>
        </div>
      </template>

      <el-empty v-else-if="!loading" description="消息不存在或已被删除">
        <el-button type="primary" @click="goBack">返回消息列表</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { useNotificationStoreHook } from "@/store/modules/notification";
import { MessageAPI, MessageVO } from "dehaze-sdk-js";
import {
  ArrowLeft,
  Bell,
  CircleCheckFilled,
  Delete,
  Position,
  Promotion,
  Star,
  WarnTriangleFilled,
} from "@element-plus/icons-vue";

defineOptions({ name: "NotifyMessageDetail" });

const route = useRoute();
const router = useRouter();
const notificationStore = useNotificationStoreHook();

const loading = ref(false);
const message = ref<MessageVO | null>(null);

const typeIconMap: Record<string, any> = {
  announcement: Bell,
  business: Promotion,
  member: Star,
  alert: WarnTriangleFilled,
  critical_alert: WarnTriangleFilled,
  inbox: Promotion,
};

function loadDetail() {
  const id = Number(route.query.id);
  if (!id) {
    message.value = null;
    return;
  }
  loading.value = true;
  MessageAPI.getDetail(id)
    .then((data) => {
      message.value = data;
      if (data.readStatus === 0) {
        MessageAPI.markRead(id).then(() => {
          message.value!.readStatus = 1;
          notificationStore.decrement();
        });
      }
    })
    .catch(() => {
      message.value = null;
    })
    .finally(() => {
      loading.value = false;
    });
}

function goBack() {
  router.push("/notify/message");
}

function handleJump() {
  if (!message.value?.jumpUrl) return;
  router.push(message.value.jumpUrl);
}

function handleDelete() {
  if (!message.value) return;
  ElMessageBox.confirm(`确定删除消息「${message.value.title}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  })
    .then(() => {
      return MessageAPI.deleteByIds(String(message.value!.id));
    })
    .then(() => {
      ElMessage.success("删除成功");
      if (message.value?.readStatus === 0) {
        notificationStore.decrement();
      }
      goBack();
    })
    .catch(() => {});
}

onMounted(() => {
  loadDetail();
});

watch(
  () => route.query.id,
  () => {
    loadDetail();
  }
);
</script>

<style lang="scss" scoped>
.message-detail {
  max-width: 800px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.detail-header {
  margin-bottom: 16px;
}

.detail-card {
  display: flex;
  overflow: hidden;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;

  .card-stripe {
    flex-shrink: 0;
    width: 5px;

    .type-announcement & {
      background: linear-gradient(180deg, #409eff, #79bbff);
    }

    .type-business & {
      background: linear-gradient(180deg, #13c2c2, #5cdbd3);
    }

    .type-member & {
      background: linear-gradient(180deg, #fa8c16, #ffc069);
    }

    .type-alert & {
      background: linear-gradient(180deg, #faad14, #ffd666);
    }

    .type-critical_alert & {
      background: linear-gradient(180deg, #f5222d, #ff7875);
    }

    .type-inbox & {
      background: linear-gradient(180deg, #8c8c8c, #bfbfbf);
    }
  }

  .card-content {
    flex: 1;
    padding: 24px 28px;
  }
}

.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 12px;

  .type-tag {
    display: inline-flex;
    gap: 4px;
    align-items: center;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 4px;

    &.tag-announcement {
      color: #409eff;
      background: #ecf5ff;
    }

    &.tag-business {
      color: #13c2c2;
      background: #e6fffb;
    }

    &.tag-member {
      color: #fa8c16;
      background: #fff7e6;
    }

    &.tag-alert {
      color: #faad14;
      background: #fffbe6;
    }

    &.tag-critical_alert {
      color: #f5222d;
      background: #fff1f0;
    }

    &.tag-inbox {
      color: #8c8c8c;
      background: #fafafa;
    }
  }

  .priority-flag {
    display: inline-flex;
    gap: 2px;
    align-items: center;
    font-size: 12px;
    font-weight: 500;
    color: #f5222d;
  }

  .sender-text {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  .time-text {
    margin-left: auto;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.detail-title {
  margin: 0 0 8px;
  font-size: 22px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--el-text-color-primary);
}

.read-info {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  font-size: 12px;
  color: var(--el-color-success);

  .el-icon {
    font-size: 14px;
  }
}

.detail-content {
  font-size: 14px;
  line-height: 1.8;
  color: var(--el-text-color-primary);
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.extra-block {
  padding: 14px 16px;
  margin-top: 20px;
  background: var(--el-fill-color-light);
  border-radius: 8px;

  .extra-title {
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 500;
    color: var(--el-text-color-secondary);
  }

  .extra-content {
    margin: 0;
    font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
    font-size: 12px;
    line-height: 1.6;
    color: var(--el-text-color-regular);
  }
}

.detail-footer {
  display: flex;
  gap: 12px;
  padding-top: 20px;
  margin-top: 28px;
  border-top: 1px solid var(--el-border-color-lighter);
}
</style>
