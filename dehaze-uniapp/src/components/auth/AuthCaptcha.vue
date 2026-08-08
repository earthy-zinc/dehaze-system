<template>
  <view class="auth-captcha">
    <view class="captcha-container">
      <view
        class="captcha-input-wrap"
        :class="{ focused: focused, error: !!error }"
      >
        <SvgIcon
          name="checkmark-circle"
          size="20"
          color="#9ca3af"
          class="input-icon"
        />
        <input
          :value="modelValue"
          class="captcha-input"
          placeholder="请输入验证码"
          placeholder-class="placeholder"
          @input="onInput"
          @focus="focused = true"
          @blur="focused = false"
        />
      </view>
      <view class="captcha-image" @click="refresh">
        <image
          v-if="captchaBase64"
          :src="captchaBase64"
          class="captcha-img"
          mode="aspectFit"
        />
        <text v-else class="captcha-placeholder">
          {{ loading ? "加载中" : "点击获取" }}
        </text>
      </view>
    </view>
    <text v-if="error" class="error-message">
      <SvgIcon name="info-circle" size="12" color="#ef4444" />
      <text>{{ error }}</text>
    </text>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import { AuthAPI } from "dehaze-sdk-js";
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  /** v-model 绑定验证码值 */
  modelValue: string;
  /** 验证码错误提示 */
  error?: string;
}

const props = withDefaults(defineProps<Props>(), {
  error: "",
});

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
}>();

const focused = ref(false);
const loading = ref(false);
const captchaBase64 = ref("");
const captchaKey = ref("");

function onInput(e: any) {
  emit("update:modelValue", e.detail.value);
}

/** 获取/刷新验证码，返回是否成功 */
async function refresh(): Promise<boolean> {
  loading.value = true;
  try {
    const result = await AuthAPI.getCaptcha();
    captchaKey.value = result.captchaKey;
    // 后端返回的 captchaBase64 已包含 "data:image/png;base64," 前缀，直接使用
    const raw = result.captchaBase64 || "";
    captchaBase64.value = raw.startsWith("data:")
      ? raw
      : `data:image/png;base64,${raw}`;
    return true;
  } catch {
    captchaBase64.value = "";
    return false;
  } finally {
    loading.value = false;
  }
}

defineExpose({
  /** 当前验证码 key（登录/注册提交时使用） */
  captchaKey: () => captchaKey.value,
  refresh,
});
</script>

<style lang="scss" scoped>
.auth-captcha {
  margin-bottom: 24rpx;
}

.captcha-container {
  display: flex;
  gap: 16rpx;
  align-items: stretch;
}

.captcha-input-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  padding: 0 24rpx;
  height: 96rpx;
  background: $color-bg-primary;
  border: 2rpx solid $color-border;
  border-radius: $radius-sm;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);

  .input-icon {
    margin-right: 16rpx;
    flex-shrink: 0;
    transition: color 0.2s ease;
  }

  .captcha-input {
    flex: 1;
    font-size: 28rpx;
    color: $color-text-primary;
    background: transparent;
    border: none;
    outline: none;
    height: 100%;
  }

  .placeholder {
    color: $color-text-placeholder;
    font-size: 28rpx;
  }

  &.focused {
    background: $color-white;
    border-color: $color-primary;
    box-shadow: $shadow-input-focus;

    .input-icon {
      color: $color-primary;
    }
  }

  &.error {
    border-color: $color-danger;
    background: $color-danger-bg;
  }
}

.captcha-image {
  width: 200rpx;
  height: 96rpx;
  border-radius: $radius-sm;
  border: 2rpx solid $color-border;
  background: $color-bg-secondary;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;

  &:active {
    border-color: $color-primary;
    background: $color-primary-bg;
  }

  .captcha-img {
    width: 100%;
    height: 100%;
  }

  .captcha-placeholder {
    font-size: 22rpx;
    color: $color-text-placeholder;
  }
}

.error-message {
  display: flex;
  align-items: center;
  gap: 6rpx;
  margin-top: 12rpx;
  font-size: 22rpx;
  color: $color-danger;
}
</style>
