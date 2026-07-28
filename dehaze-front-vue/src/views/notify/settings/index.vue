<template>
  <div class="app-container notification-settings" v-loading="loading">
    <div class="page-header">
      <div class="header-title">
        <el-button link @click="router.push('/notify/message')">
          <el-icon><ArrowLeft /></el-icon>
        </el-button>
        <span class="title-text">通知设置</span>
      </div>
      <el-button
        :disabled="!hasChange"
        :loading="saving"
        type="primary"
        @click="handleSave"
      >
        <el-icon><Check /></el-icon>
        保存设置
      </el-button>
    </div>

    <template v-if="formData">
      <!-- 推送总开关 -->
      <section class="settings-section">
        <div class="section-header">
          <div class="section-title">
            <el-icon class="section-icon"><Bell /></el-icon>
            <span>推送通知</span>
          </div>
          <span class="section-desc">控制是否接收 APP 推送通知</span>
        </div>
        <div class="section-body">
          <div class="switch-row">
            <div class="switch-label">
              <span class="label-text">APP 推送</span>
              <span class="label-desc">开启后将通过 APP 推送接收消息通知</span>
            </div>
            <el-switch v-model="formData.pushEnabled" />
          </div>
        </div>
      </section>

      <!-- 免打扰设置 -->
      <section class="settings-section">
        <div class="section-header">
          <div class="section-title">
            <el-icon class="section-icon"><Moon /></el-icon>
            <span>免打扰</span>
          </div>
          <span class="section-desc">在指定时段内不接收推送通知</span>
        </div>
        <div class="section-body">
          <div class="switch-row">
            <div class="switch-label">
              <span class="label-text">开启免打扰</span>
              <span class="label-desc">在指定时段内静默所有推送</span>
            </div>
            <el-switch v-model="formData.dndEnabled" />
          </div>
          <div v-if="formData.dndEnabled" class="dnd-time-row">
            <div class="time-picker-item">
              <label>开始时间</label>
              <el-time-picker
                v-model="dndStartDate"
                format="HH:mm"
                value-format="HH:mm:ss"
                placeholder="开始时间"
              />
            </div>
            <div class="time-separator">至</div>
            <div class="time-picker-item">
              <label>结束时间</label>
              <el-time-picker
                v-model="dndEndDate"
                format="HH:mm"
                value-format="HH:mm:ss"
                placeholder="结束时间"
              />
            </div>
          </div>
        </div>
      </section>

      <!-- 按消息类型设置推送 -->
      <section v-if="formData.preferences" class="settings-section">
        <div class="section-header">
          <div class="section-title">
            <el-icon class="section-icon"><Operation /></el-icon>
            <span>按类型设置推送</span>
          </div>
          <span class="section-desc">为不同类型的消息单独配置推送开关</span>
        </div>
        <div class="section-body">
          <div
            v-for="item in typeChannelList"
            :key="item.key"
            class="switch-row"
          >
            <div class="switch-label">
              <span :class="['label-text', 'type-label', `type-${item.key}`]">
                <span class="type-dot"></span>
                {{ item.label }}
              </span>
              <span class="label-desc">{{ item.desc }}</span>
            </div>
            <el-switch
              v-model="formData.preferences.typeChannels[item.key].push"
            />
          </div>
        </div>
      </section>

      <!-- 模块通知开关 -->
      <section v-if="formData.preferences" class="settings-section">
        <div class="section-header">
          <div class="section-title">
            <el-icon class="section-icon"><Grid /></el-icon>
            <span>模块通知</span>
          </div>
          <span class="section-desc">按业务模块开关对应的通知</span>
        </div>
        <div class="section-body">
          <div
            v-for="item in moduleSwitchList"
            :key="item.key"
            class="switch-row"
          >
            <div class="switch-label">
              <span class="label-text">{{ item.label }}</span>
              <span class="label-desc">{{ item.desc }}</span>
            </div>
            <el-switch
              v-model="formData.preferences.moduleSwitches[item.key]"
            />
          </div>
          <div v-if="moduleSwitchList.length === 0" class="empty-tip">
            暂无可配置的模块
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<script lang="ts" setup>
import { NotificationSettingAPI, NotificationSettings } from "dehaze-sdk-js";
import {
  ArrowLeft,
  Bell,
  Check,
  Grid,
  Moon,
  Operation,
} from "@element-plus/icons-vue";

defineOptions({ name: "NotifySettings" });

const router = useRouter();

const loading = ref(false);
const saving = ref(false);
const formData = ref<NotificationSettings | null>(null);
const originalSnapshot = ref<string>("");
const dndStartDate = ref<Date | string>("");
const dndEndDate = ref<Date | string>("");

const typeChannelList = [
  {
    key: "announcement",
    label: "系统公告",
    desc: "系统维护、功能更新、活动通知等",
  },
  {
    key: "business",
    label: "业务通知",
    desc: "订单状态变更、任务完成、反馈回复等",
  },
  { key: "member", label: "会员通知", desc: "等级变更、到期预警、权益更新等" },
];

const moduleSwitchList = [
  { key: "prediction", label: "去雾处理", desc: "去雾任务完成、失败等通知" },
  { key: "feedback", label: "反馈评价", desc: "反馈回复、评价处理等通知" },
  { key: "announcement", label: "系统公告", desc: "管理员发送的系统公告" },
];

const hasChange = computed(() => {
  if (!formData.value) return false;
  return JSON.stringify(buildPayload()) !== originalSnapshot.value;
});

function buildPayload() {
  if (!formData.value) return {};
  return {
    pushEnabled: formData.value.pushEnabled,
    dndEnabled: formData.value.dndEnabled,
    dndStart: typeof dndStartDate.value === "string" ? dndStartDate.value : "",
    dndEnd: typeof dndEndDate.value === "string" ? dndEndDate.value : "",
    preferences: formData.value.preferences,
  };
}

function loadSettings() {
  loading.value = true;
  NotificationSettingAPI.get()
    .then((data) => {
      formData.value = {
        pushEnabled: data.pushEnabled,
        dndEnabled: data.dndEnabled,
        dndStart: data.dndStart,
        dndEnd: data.dndEnd,
        preferences: {
          typeChannels: data.preferences?.typeChannels ?? {
            announcement: { push: true },
            business: { push: true },
            member: { push: true },
          },
          moduleSwitches: data.preferences?.moduleSwitches ?? {
            prediction: true,
            feedback: true,
            announcement: true,
          },
        },
      };
      dndStartDate.value = data.dndStart;
      dndEndDate.value = data.dndEnd;
      originalSnapshot.value = JSON.stringify(buildPayload());
    })
    .finally(() => {
      loading.value = false;
    });
}

function handleSave() {
  if (!formData.value) return;
  saving.value = true;
  NotificationSettingAPI.update(buildPayload())
    .then(() => {
      ElMessage.success("设置已保存");
      originalSnapshot.value = JSON.stringify(buildPayload());
    })
    .finally(() => {
      saving.value = false;
    });
}

onMounted(() => {
  loadSettings();
});
</script>

<style lang="scss" scoped>
.notification-settings {
  max-width: 720px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;

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

.settings-section {
  margin-bottom: 20px;
  overflow: hidden;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;

  .section-header {
    padding: 16px 20px;
    background: var(--el-fill-color-lighter);
    border-bottom: 1px solid var(--el-border-color-lighter);

    .section-title {
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 15px;
      font-weight: 600;
      color: var(--el-text-color-primary);

      .section-icon {
        font-size: 18px;
        color: var(--el-color-primary);
      }
    }

    .section-desc {
      display: block;
      margin-top: 4px;
      margin-left: 26px;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .section-body {
    padding: 8px 20px;
  }
}

.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);

  &:last-child {
    border-bottom: none;
  }

  .switch-label {
    display: flex;
    flex-direction: column;
    gap: 2px;

    .label-text {
      font-size: 14px;
      font-weight: 500;
      color: var(--el-text-color-primary);
    }

    .label-desc {
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }

    .type-label {
      display: inline-flex;
      gap: 6px;
      align-items: center;

      .type-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
      }

      &.type-announcement .type-dot {
        background: #409eff;
      }

      &.type-business .type-dot {
        background: #13c2c2;
      }

      &.type-member .type-dot {
        background: #fa8c16;
      }
    }
  }
}

.dnd-time-row {
  display: flex;
  gap: 16px;
  align-items: flex-end;
  padding: 16px 0 12px;
  margin-top: 8px;
  border-top: 1px dashed var(--el-border-color-lighter);

  .time-picker-item {
    display: flex;
    flex-direction: column;
    gap: 6px;

    label {
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .time-separator {
    padding-bottom: 8px;
    font-size: 14px;
    color: var(--el-text-color-secondary);
  }
}

.empty-tip {
  padding: 20px 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  text-align: center;
}
</style>
