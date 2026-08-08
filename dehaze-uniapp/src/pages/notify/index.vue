<template>
  <PageLayout level="L2" title="消息设置" class="page">
    <view class="main-content">
      <!-- 推送通知总开关 -->
      <view class="setting-group">
        <text class="group-title">通知开关</text>
        <view class="setting-card">
          <view class="setting-item">
            <view class="setting-item-row">
              <text class="setting-icon">📢</text>
              <view>
                <text class="setting-label">推送通知</text>
                <text class="setting-desc">接收所有类型的推送通知</text>
              </view>
            </view>
            <switch
              :checked="settings.pushEnabled"
              @change="(e: any) => togglePushEnabled(e.detail.value)"
              color="#3b82f6"
            />
          </view>
          <view
            v-for="mod in moduleSwitches"
            :key="mod.key"
            class="setting-item"
          >
            <view class="setting-item-row">
              <text class="setting-icon">{{ mod.icon }}</text>
              <view>
                <text class="setting-label">{{ mod.label }}</text>
                <text class="setting-desc">{{ mod.desc }}</text>
              </view>
            </view>
            <switch
              :checked="getModuleSwitch(mod.key)"
              @change="(e: any) => toggleModuleSwitch(mod.key, e.detail.value)"
              color="#3b82f6"
            />
          </view>
        </view>
      </view>

      <!-- 免打扰设置 -->
      <view class="setting-group">
        <text class="group-title">免打扰</text>
        <view class="setting-card">
          <view class="setting-item">
            <view class="setting-item-row">
              <text class="setting-icon">🌙</text>
              <view>
                <text class="setting-label">免打扰模式</text>
                <text class="setting-desc">开启后在设定时段内不接收通知</text>
              </view>
            </view>
            <switch
              :checked="settings.dndEnabled"
              @change="(e: any) => toggleDndEnabled(e.detail.value)"
              color="#8b5cf6"
            />
          </view>
          <view v-if="settings.dndEnabled" class="setting-item">
            <text class="setting-label">开始时间</text>
            <picker
              mode="selector"
              :range="timeOptions"
              :value="dndStartIdx"
              @change="handleDndStartChange"
            >
              <text class="setting-value">{{ settings.dndStart || "22:00" }}</text>
            </picker>
          </view>
          <view v-if="settings.dndEnabled" class="setting-item">
            <text class="setting-label">结束时间</text>
            <picker
              mode="selector"
              :range="timeOptions"
              :value="dndEndIdx"
              @change="handleDndEndChange"
            >
              <text class="setting-value">{{ settings.dndEnd || "08:00" }}</text>
            </picker>
          </view>
        </view>
      </view>

      <view class="notify-footer">
        <text class="footer-text">设置实时生效</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { NotificationSettingAPI } from "dehaze-sdk-js";

const moduleSwitches = [
  { key: "system", label: "系统通知", desc: "接收系统公告和重要通知", icon: "🔔" },
  { key: "business", label: "业务通知", desc: "接收业务流程相关通知", icon: "📋" },
  { key: "member", label: "会员通知", desc: "接收会员权益和到期提醒", icon: "👑" },
  { key: "activity", label: "活动通知", desc: "接收优惠活动和促销信息", icon: "🎉" },
];

const timeOptions: string[] = [];
for (let h = 0; h < 24; h++) {
  for (let m = 0; m < 60; m += 30) {
    timeOptions.push(`${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`);
  }
}

const settings = reactive({
  pushEnabled: true,
  dndEnabled: false,
  dndStart: "22:00",
  dndEnd: "08:00",
  preferences: {
    typeChannels: {} as Record<string, { push: boolean }>,
    moduleSwitches: {} as Record<string, boolean>,
  },
});

const saving = ref(false);
const dndStartIdx = ref<number[]>([44]); // 22:00
const dndEndIdx = ref<number[]>([16]);   // 08:00

function getModuleSwitch(module: string): boolean {
  return settings.preferences.moduleSwitches[module] !== false;
}

async function doSave(data: Record<string, unknown>) {
  if (saving.value) return;
  saving.value = true;
  try {
    await NotificationSettingAPI.update(data as any);
    uni.showToast({ title: "已保存", icon: "success" });
  } catch {
    uni.showToast({ title: "保存失败", icon: "none" });
  } finally {
    saving.value = false;
  }
}

function togglePushEnabled(val: boolean) {
  settings.pushEnabled = val;
  doSave({ pushEnabled: val });
}

function toggleModuleSwitch(module: string, val: boolean) {
  settings.preferences.moduleSwitches[module] = val;
  doSave({ preferences: { moduleSwitches: { [module]: val } } });
}

function toggleDndEnabled(val: boolean) {
  settings.dndEnabled = val;
  doSave({ dndEnabled: val });
}

function handleDndStartChange(e: { detail: { value: number } }) {
  const idx = e.detail.value;
  dndStartIdx.value = [idx];
  const time = timeOptions[idx];
  settings.dndStart = time;
  doSave({ dndStart: time });
}

function handleDndEndChange(e: { detail: { value: number } }) {
  const idx = e.detail.value;
  dndEndIdx.value = [idx];
  const time = timeOptions[idx];
  settings.dndEnd = time;
  doSave({ dndEnd: time });
}

onMounted(async () => {
  try {
    const data = await NotificationSettingAPI.get();
    if (data) {
      settings.pushEnabled = data.pushEnabled;
      settings.dndEnabled = data.dndEnabled;
      settings.dndStart = data.dndStart || "22:00";
      settings.dndEnd = data.dndEnd || "08:00";
      settings.preferences = data.preferences || {
        typeChannels: {},
        moduleSwitches: {},
      };
      if (data.dndStart) {
        const idx = timeOptions.indexOf(data.dndStart);
        if (idx >= 0) dndStartIdx.value = [idx];
      }
      if (data.dndEnd) {
        const idx = timeOptions.indexOf(data.dndEnd);
        if (idx >= 0) dndEndIdx.value = [idx];
      }
    }
  } catch {
    // use defaults
  }
});
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}
.main-content {
  padding: $spacing-md;
}
.setting-group {
  margin-bottom: $spacing-md;
}
.group-title {
  display: block;
  font-size: $font-xs;
  font-weight: 500;
  color: $color-text-placeholder;
  padding: 0 4rpx 12rpx;
}
.setting-card {
  background: #fff;
  border-radius: $radius-xl;
  overflow: hidden;
  box-shadow: $shadow-sm;
}
.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 28rpx;
  & + & {
    border-top: 1rpx solid $color-border-light;
  }
}
.setting-item-row {
  display: flex;
  gap: 16rpx;
  align-items: center;
}
.setting-icon {
  font-size: 36rpx;
}
.setting-label {
  display: block;
  font-size: $font-md;
  color: $color-text-primary;
}
.setting-desc {
  display: block;
  font-size: $font-xs;
  color: $color-text-secondary;
  margin-top: 4rpx;
}
.setting-value {
  font-size: $font-md;
  color: $color-primary;
}
.notify-footer {
  padding: 32rpx 0 48rpx;
  text-align: center;
}
.footer-text {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
</style>
