<template>
  <AuthShell title="欢迎回来" subtitle="让每一帧画面回归清晰">
    <AuthInput
      v-model="formData.username"
      icon="account"
      placeholder="用户名"
    />

    <AuthInput
      v-model="formData.password"
      icon="lock"
      password
      placeholder="密码"
    />

    <AuthCaptcha
      v-model="formData.captcha"
      :error="captchaError"
      ref="captchaRef"
    />

    <!-- 记住我 + 忘记密码 -->
    <view class="option-row">
      <view class="remember-me" @click="rememberMe = !rememberMe">
        <view class="remember-checkbox" :class="{ checked: rememberMe }">
          <SvgIcon
            v-if="rememberMe"
            name="checkmark"
            size="14"
            color="#ffffff"
          />
        </view>
        <text class="remember-text">记住我</text>
      </view>
      <text class="forgot-link" @click="handleForgot">忘记密码？</text>
    </view>

    <button
      :disabled="loading"
      class="submit-button"
      :class="{ loading }"
      @click="handleSubmit"
    >
      <view v-if="loading" class="loading-spinner"></view>
      <text>{{ loading ? "登录中..." : "登 录" }}</text>
    </button>

    <!-- 注册入口 -->
    <view class="register-row">
      <text class="register-text">还没有账号？</text>
      <text class="register-link" @click="goRegister">立即注册</text>
    </view>
  </AuthShell>
</template>

<script lang="ts" setup>
import { reactive, ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import AuthShell from "@/components/auth/AuthShell.vue";
import AuthInput from "@/components/auth/AuthInput.vue";
import AuthCaptcha from "@/components/auth/AuthCaptcha.vue";
import { useAuthStore } from "@/store/auth";
import { HOME_PATH } from "@/routers/guard";
import { getErrorMessage } from "@/utils/error";

const loading = ref(false);
const rememberMe = ref(false);
const captchaError = ref("");
const authStore = useAuthStore();
const captchaRef = ref<InstanceType<typeof AuthCaptcha>>();

const formData = reactive({
  username: "",
  password: "",
  captcha: "",
});

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
      captchaKey: captchaRef.value?.captchaKey() || "",
      captchaCode: formData.captcha,
      rememberMe: rememberMe.value,
    });

    // 登录成功立即进入首页，首页自然呈现登录态
    uni.switchTab({ url: HOME_PATH });
  } catch (error) {
    const errorMsg = getErrorMessage(error, "登录失败，请重试");

    // 验证码错误时刷新验证码
    if (errorMsg.includes("验证码")) {
      captchaError.value = errorMsg;
      captchaRef.value?.refresh();
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

/** 忘记密码（当前无找回密码模块） */
const handleForgot = () => {
  uni.showToast({ title: "请联系管理员重置密码", icon: "none" });
};

/** 跳转注册 */
const goRegister = () => {
  uni.navigateTo({ url: "/pages/register/index" });
};

onMounted(async () => {
  // 已登录直接进入首页（含登录失效被重定向后重新登录的场景）
  if (authStore.isLoggedIn) {
    uni.switchTab({ url: HOME_PATH });
    return;
  }
  // 自动获取验证码
  await captchaRef.value?.refresh();
});
</script>

<style lang="scss" scoped>
.option-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 32rpx;
}

.remember-me {
  display: flex;
  align-items: center;
  gap: 12rpx;

  .remember-checkbox {
    width: 36rpx;
    height: 36rpx;
    border-radius: 8rpx;
    border: 2rpx solid $color-border;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;

    &.checked {
      background: $color-primary;
      border-color: $color-primary;
    }
  }

  .remember-text {
    font-size: 26rpx;
    color: $color-text-secondary;
  }
}

.forgot-link {
  font-size: 26rpx;
  color: $color-primary;
}

.submit-button {
  width: 100%;
  height: 96rpx;
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
    background: linear-gradient(
      135deg,
      $color-primary-light 0%,
      $color-secondary-light 100%
    );
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

.register-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  margin-top: 40rpx;

  .register-text {
    font-size: 26rpx;
    color: $color-text-secondary;
  }

  .register-link {
    font-size: 26rpx;
    font-weight: 500;
    color: $color-primary;
  }
}
</style>
