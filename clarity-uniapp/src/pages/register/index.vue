<template>
  <AuthShell title="创建账号" subtitle="加入图像去雾，开启清晰之旅">
    <AuthInput
      v-model="form.username"
      icon="account"
      placeholder="用户名（3-32位字母数字下划线）"
    />

    <AuthInput v-model="form.nickname" icon="account-fill" placeholder="昵称" />

    <AuthInput
      v-model="form.password"
      icon="lock"
      password
      placeholder="密码（6-20位含字母和数字）"
    />

    <AuthInput
      v-model="form.confirmPassword"
      icon="lock"
      password
      placeholder="确认密码"
    />

    <AuthCaptcha
      v-model="form.captchaCode"
      :error="captchaError"
      ref="captchaRef"
    />

    <!-- 协议勾选 -->
    <view class="agreement-row" @click="agreed = !agreed">
      <view class="agreement-checkbox" :class="{ checked: agreed }">
        <SvgIcon v-if="agreed" name="checkmark" size="14" color="#ffffff" />
      </view>
      <text class="agreement-text">
        我已阅读并同意《用户协议》和《隐私政策》
      </text>
    </view>

    <button
      :disabled="loading"
      class="submit-button"
      :class="{ loading }"
      @click="handleRegister"
    >
      <view v-if="loading" class="loading-spinner"></view>
      <text>{{ loading ? "注册中..." : "注 册" }}</text>
    </button>

    <!-- 登录入口 -->
    <view class="login-row">
      <text class="login-text">已有账号？</text>
      <text class="login-link" @click="goLogin">立即登录</text>
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
import { HOME_PATH, LOGIN_PATH } from "@/routers/guard";
import { getErrorMessage } from "@/utils/error";

const loading = ref(false);
const agreed = ref(false);
const captchaError = ref("");
const authStore = useAuthStore();
const captchaRef = ref<InstanceType<typeof AuthCaptcha>>();

const form = reactive({
  username: "",
  nickname: "",
  password: "",
  confirmPassword: "",
  captchaCode: "",
});

/** 表单校验 */
const validateForm = (): boolean => {
  captchaError.value = "";

  if (!form.username.trim()) {
    uni.showToast({ title: "请输入用户名", icon: "none" });
    return false;
  }
  if (!/^[a-zA-Z0-9_]{3,32}$/.test(form.username.trim())) {
    uni.showToast({ title: "用户名需为3-32位字母数字下划线", icon: "none" });
    return false;
  }
  if (!form.nickname.trim()) {
    uni.showToast({ title: "请输入昵称", icon: "none" });
    return false;
  }
  if (!form.password) {
    uni.showToast({ title: "请输入密码", icon: "none" });
    return false;
  }
  if (!/^(?=.*[a-zA-Z])(?=.*\d).{6,20}$/.test(form.password)) {
    uni.showToast({ title: "密码需为6-20位且包含字母和数字", icon: "none" });
    return false;
  }
  if (form.password !== form.confirmPassword) {
    uni.showToast({ title: "两次密码不一致", icon: "none" });
    return false;
  }
  if (!form.captchaCode.trim()) {
    captchaError.value = "请输入验证码";
    return false;
  }
  if (!agreed.value) {
    uni.showToast({ title: "请先阅读并同意用户协议和隐私政策", icon: "none" });
    return false;
  }

  return true;
};

/** 提交注册：成功后自动登录进入首页 */
const handleRegister = async () => {
  if (!validateForm()) return;

  loading.value = true;

  try {
    await authStore.register({
      username: form.username.trim(),
      password: form.password,
      nickname: form.nickname.trim(),
      captchaKey: captchaRef.value?.captchaKey() || "",
      captchaCode: form.captchaCode,
    });

    uni.switchTab({ url: HOME_PATH });
  } catch (error) {
    const errorMsg = getErrorMessage(error, "注册失败，请重试");

    // 验证码错误时刷新验证码
    if (errorMsg.includes("验证码")) {
      captchaError.value = errorMsg;
      captchaRef.value?.refresh();
      form.captchaCode = "";
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

/** 返回登录 */
const goLogin = () => {
  uni.navigateBack({
    fail: () => uni.reLaunch({ url: LOGIN_PATH }),
  });
};

onMounted(async () => {
  // 已登录直接进入首页
  if (authStore.isLoggedIn) {
    uni.switchTab({ url: HOME_PATH });
    return;
  }
  // 自动获取验证码
  await captchaRef.value?.refresh();
});
</script>

<style lang="scss" scoped>
.submit-button {
  width: 100%;
  height: 96rpx;
  margin-top: 8rpx;
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

.agreement-row {
  display: flex;
  align-items: flex-start;
  gap: 12rpx;
  margin-bottom: 8rpx;

  .agreement-checkbox {
    width: 36rpx;
    height: 36rpx;
    border-radius: 8rpx;
    border: 2rpx solid $color-border;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2rpx;
    transition: all 0.2s ease;

    &.checked {
      background: $color-primary;
      border-color: $color-primary;
    }
  }

  .agreement-text {
    font-size: 24rpx;
    color: $color-text-secondary;
    line-height: 1.6;
  }
}

.agreement-row {
  display: flex;
  align-items: flex-start;
  gap: 12rpx;
  margin-bottom: 8rpx;

  .agreement-checkbox {
    width: 36rpx;
    height: 36rpx;
    border-radius: 8rpx;
    border: 2rpx solid $color-border;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2rpx;
    transition: all 0.2s ease;

    &.checked {
      background: $color-primary;
      border-color: $color-primary;
    }
  }

  .agreement-text {
    font-size: 24rpx;
    color: $color-text-secondary;
    line-height: 1.6;
  }
}

.login-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  margin-top: 40rpx;

  .login-text {
    font-size: 26rpx;
    color: $color-text-secondary;
  }

  .login-link {
    font-size: 26rpx;
    font-weight: 500;
    color: $color-primary;
  }
}
</style>
