<template>
  <view class="login-page">
    <!-- 顶部渐变背景区 -->
    <view class="hero-bg">
      <view class="hero-decoration hero-decoration-1"></view>
      <view class="hero-decoration hero-decoration-2"></view>

      <view class="brand-area">
        <view class="brand-logo">
          <u-icon name="photo-fill" size="44" color="#ffffff" />
        </view>
        <text class="brand-title">图像去雾系统</text>
        <text class="brand-subtitle">专业级图像处理平台</text>
      </view>
    </view>

    <!-- 登录卡片 -->
    <view class="login-card">
      <view class="card-header">
        <text class="card-title">欢迎登录</text>
        <text class="card-subtitle">请使用您的账号登录系统</text>
      </view>

      <form class="form-container" @submit.prevent="handleSubmit">
        <view class="form-group">
          <view
            class="input-wrapper"
            :class="{ focused: focusedField === 'username' }"
          >
            <u-icon
              name="account"
              size="20"
              color="#9ca3af"
              class="input-icon"
            />
            <input
              v-model="formData.username"
              class="form-input"
              placeholder="请输入用户名"
              placeholder-class="placeholder"
              @focus="focusedField = 'username'"
              @blur="focusedField = ''"
            />
          </view>
        </view>

        <view class="form-group">
          <view
            class="input-wrapper"
            :class="{ focused: focusedField === 'password' }"
          >
            <u-icon name="lock" size="20" color="#9ca3af" class="input-icon" />
            <input
              v-model="formData.password"
              class="form-input"
              password
              placeholder="请输入密码"
              placeholder-class="placeholder"
              @focus="focusedField = 'password'"
              @blur="focusedField = ''"
            />
          </view>
        </view>

        <view class="form-group">
          <view class="captcha-container">
            <view
              class="input-wrapper captcha-input"
              :class="{
                focused: focusedField === 'captcha',
                error: !!captchaError,
              }"
            >
              <u-icon
                name="checkmark-circle"
                size="20"
                color="#9ca3af"
                class="input-icon"
              />
              <input
                v-model="formData.captcha"
                class="form-input"
                placeholder="请输入验证码"
                placeholder-class="placeholder"
                @focus="focusedField = 'captcha'"
                @blur="focusedField = ''"
              />
            </view>
            <view class="captcha-image" @click="refreshCaptcha">
              <image
                v-if="captchaBase64"
                :src="captchaBase64"
                class="captcha-img"
                mode="aspectFit"
              />
              <text v-else class="captcha-placeholder">点击获取</text>
            </view>
          </view>
          <text v-if="captchaError" class="error-message">
            <u-icon name="info-circle" size="12" color="#ef4444" />
            <text>{{ captchaError }}</text>
          </text>
        </view>

        <button
          :disabled="loading"
          class="submit-button"
          :class="{ loading: loading }"
          @click="handleSubmit"
        >
          <view v-if="loading" class="loading-spinner"></view>
          <text>{{ loading ? "登录中..." : "登 录" }}</text>
        </button>

        <view class="hint-area">
          <u-icon name="info-circle" size="12" color="#9ca3af" />
          <text class="hint-text">请输入账号和密码登录</text>
        </view>
      </form>
    </view>

    <!-- 底部信息 -->
    <view class="login-footer">
      <text class="footer-text">Copyright © 2022 - 2024 Peixin Wu</text>
      <text class="footer-text">渝ICP备2024111923号-2</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { reactive, ref, onMounted } from "vue";
import { useAuthStore } from "@/store/auth";
import { navigateToHome } from "@/routers/guard";
import { getErrorMessage } from "@/utils/error";

// ==================== 状态定义 ====================

const loading = ref(false);
const authStore = useAuthStore();
const focusedField = ref("");

const formData = reactive({
  username: "",
  password: "",
  captcha: "",
});

/** 验证码缓存 key（登录时需要传回） */
const captchaKey = ref("");

/** 验证码图片 Base64 */
const captchaBase64 = ref("");

/** 验证码错误提示 */
const captchaError = ref("");

// ==================== 方法定义 ====================

/** 获取验证码 */
const refreshCaptcha = async () => {
  captchaError.value = "";
  captchaBase64.value = "";

  try {
    const result = await authStore.getCaptcha();
    captchaKey.value = result.captchaKey;
    // 后端返回的 captchaBase64 已包含 "data:image/png;base64," 前缀，直接使用
    const raw = result.captchaBase64 || "";
    captchaBase64.value = raw.startsWith("data:")
      ? raw
      : `data:image/png;base64,${raw}`;
  } catch {
    captchaError.value = "获取验证码失败，请重试";
  }
};

/** 表单校验 */
const validateForm = (): boolean => {
  captchaError.value = "";

  if (!formData.username.trim()) {
    uni.showToast({ title: "请输入用户名", icon: "none" });
    return false;
  }
  if (!formData.password.trim()) {
    uni.showToast({ title: "请输入密码", icon: "none" });
    return false;
  }
  if (!formData.captcha.trim()) {
    captchaError.value = "请输入验证码";
    return false;
  }

  return true;
};

/** 提交登录 */
const handleSubmit = async () => {
  if (!validateForm()) return;

  loading.value = true;

  try {
    await authStore.login({
      username: formData.username,
      password: formData.password,
      captchaKey: captchaKey.value,
      captchaCode: formData.captcha,
    });

    uni.showToast({
      title: "登录成功",
      icon: "success",
      duration: 1500,
    });

    // 延迟跳转，让用户看到成功提示
    setTimeout(() => {
      navigateToHome();
    }, 1500);
  } catch (error) {
    const errorMsg = getErrorMessage(error, "登录失败，请重试");

    // 验证码错误时刷新验证码
    if (errorMsg.includes("验证码")) {
      captchaError.value = errorMsg;
      refreshCaptcha();
      formData.captcha = "";
    } else {
      uni.showToast({
        title: errorMsg,
        icon: "none",
        duration: 2500,
      });
    }
  } finally {
    loading.value = false;
  }
};

// ==================== 生命周期 ====================

onMounted(() => {
  // 如果已登录，直接跳转首页
  if (authStore.isLoggedIn) {
    navigateToHome();
    return;
  }
  // 自动获取验证码
  refreshCaptcha();
});
</script>

<style lang="scss" scoped>
/* ==================== 设计变量 ==================== */
/* 使用全局令牌：通过 vite additionalData 自动注入 @/styles/variables.scss */
/* 品牌色、文字色、圆角、阴影等令牌均来自 variables.scss，无需在此重复定义 */

.login-page {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: $color-bg-primary;
  font-family:
    -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;

  /* #ifdef H5 */
  min-height: calc(100vh - 44px - env(safe-area-inset-top));
  /* #endif */
}

/* ==================== 顶部渐变背景区 ==================== */
.hero-bg {
  position: relative;
  height: 320rpx;
  background: $gradient-primary;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;

  .hero-decoration {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    pointer-events: none;

    &-1 {
      width: 400rpx;
      height: 400rpx;
      top: -200rpx;
      right: -100rpx;
    }

    &-2 {
      width: 300rpx;
      height: 300rpx;
      bottom: -150rpx;
      left: -80rpx;
      background: rgba(255, 255, 255, 0.06);
    }
  }
}

.brand-area {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;

  .brand-logo {
    width: 120rpx;
    height: 120rpx;
    border-radius: 28rpx;
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(20rpx);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20rpx;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }

  .brand-title {
    font-size: 40rpx;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2rpx;
    margin-bottom: 8rpx;
    text-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.1);
  }

  .brand-subtitle {
    font-size: 24rpx;
    color: rgba(255, 255, 255, 0.85);
    letter-spacing: 1rpx;
  }
}

/* ==================== 登录卡片 ==================== */
.login-card {
  position: relative;
  z-index: 2;
  margin: -60rpx 40rpx 0;
  background: $color-white;
  border-radius: $radius-lg;
  box-shadow: $shadow-lg;
  padding: 48rpx 40rpx 36rpx;
  animation: cardEnter 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes cardEnter {
  from {
    opacity: 0;
    transform: translateY(20rpx);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.card-header {
  text-align: center;
  margin-bottom: 40rpx;

  .card-title {
    display: block;
    font-size: 36rpx;
    font-weight: 600;
    color: $color-text-primary;
    margin-bottom: 8rpx;
    letter-spacing: 1rpx;
  }

  .card-subtitle {
    display: block;
    font-size: 24rpx;
    color: $color-text-secondary;
  }
}

/* ==================== 表单 ==================== */
.form-container {
  .form-group {
    margin-bottom: 24rpx;
  }

  .input-wrapper {
    display: flex;
    align-items: center;
    padding: 0 24rpx;
    height: 88rpx;
    background: $color-bg-primary;
    border: 2rpx solid $color-border;
    border-radius: $radius-sm;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);

    .input-icon {
      margin-right: 16rpx;
      flex-shrink: 0;
      transition: color 0.2s ease;
    }

    .form-input {
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

  .captcha-container {
    display: flex;
    gap: 16rpx;
    align-items: stretch;

    .captcha-input {
      flex: 1;
    }

    .captcha-image {
      width: 180rpx;
      height: 88rpx;
      border-radius: $radius-sm;
      border: 2rpx solid $color-border;
      background: $color-bg-secondary;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.2s ease;

      &:hover {
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
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 6rpx;
    margin-top: 12rpx;
    font-size: 22rpx;
    color: $color-danger;
  }
}

/* ==================== 提交按钮 ==================== */
.submit-button {
  width: 100%;
  height: 96rpx;
  margin-top: 16rpx;
  border: none;
  border-radius: $radius-md;
  background: $gradient-primary;
  background-size: 200% auto;
  color: $color-white;
  font-size: 32rpx;
  font-weight: 600;
  letter-spacing: 8rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12rpx;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 8rpx 20rpx rgba(59, 130, 246, 0.28);

  &:active {
    transform: scale(0.98);
    box-shadow: 0 4rpx 12rpx rgba(59, 130, 246, 0.24);
  }

  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }

  &.loading {
    background: linear-gradient(135deg, $color-primary-light 0%, $color-secondary-light 100%);
  }
}

.loading-spinner {
  width: 28rpx;
  height: 28rpx;
  border: 3rpx solid rgba(255, 255, 255, 0.3);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ==================== 提示区 ==================== */
.hint-area {
  margin-top: 28rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;

  .hint-text {
    font-size: 22rpx;
    color: $color-text-placeholder;
  }
}

/* ==================== 底部 ==================== */
.login-footer {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  padding: 40rpx 0 48rpx;
  text-align: center;

  .footer-text {
    font-size: 20rpx;
    color: $color-text-placeholder;
    line-height: 1.8;
  }
}
</style>
