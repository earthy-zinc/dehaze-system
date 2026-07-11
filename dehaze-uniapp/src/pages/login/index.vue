<template>
  <view class="login-container">
    <view class="login-header"></view>

    <view class="login-card">
      <view class="login-header">
        <text class="app-title">图像去雾系统</text>
        <text class="version">1.10.1</text>
      </view>

      <form class="form-container" @submit.prevent="handleSubmit">
        <view class="form-group">
          <input
            v-model="formData.username"
            class="form-input"
            placeholder="请输入用户名"
          />
        </view>

        <view class="form-group">
          <input
            v-model="formData.password"
            class="form-input"
            password
            placeholder="请输入密码"
          />
        </view>

        <view class="form-group">
          <view class="captcha-container">
            <input
              v-model="formData.captcha"
              :class="{ 'form-input': true, error: captchaError }"
              placeholder="请输入验证码"
            />
            <view class="captcha-image" @click="refreshCaptcha">
              <image
                v-if="captchaBase64"
                :src="captchaBase64"
                class="captcha-img"
                mode="aspectFit"
              />
              <text v-else class="captcha-text">点击获取</text>
            </view>
          </view>
          <text v-if="captchaError" class="error-message">
            {{ captchaError }}
          </text>
        </view>

        <button :disabled="loading" class="form-button" @click="handleSubmit">
          {{ loading ? "登录中..." : "登 录" }}
        </button>

        <view class="footer-info">
          <text class="info-text">用户名: admin</text>
          <text class="info-text">密码: 123456</text>
        </view>
      </form>
    </view>

    <view class="login-footer">
      <view class="login-footer-text">
        <text>Copyright © 2022 - 2024 Peixin Wu All Rights Reserved.</text>
      </view>
      <view class="login-footer-text">
        <text>武沛鑫 版权所有</text>
      </view>
      <view class="login-footer-text">
        <text>渝ICP备2024111923号-2</text>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { reactive, ref, onMounted } from "vue";
import { useAuthStore } from "@/store/auth";
import { navigateToHome } from "@/routers/guard";

// ==================== 状态定义 ====================

const loading = ref(false);
const authStore = useAuthStore();

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
    captchaBase64.value = `data:image/png;base64,${result.captchaBase64}`;
  } catch (error) {
    captchaError.value = "获取验证码失败，请重试";
    console.error("[Login] 获取验证码失败:", error);
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
    const err = error as { msg?: string; message?: string };
    const errorMsg = err?.msg || err?.message || "登录失败，请重试";

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
.login-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  height: 100vh;

  /* #ifdef H5 */
  height: calc(100vh - 44px - env(safe-area-inset-top));
  /* #endif */

  background-color: #f5f7fa;

  .login-header {
    margin-top: 32px;
  }

  .login-card {
    width: 80vw;
    max-width: 340px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 10px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    padding: 12px 24px;
    transition: all 0.3s ease;

    .login-header {
      display: flex;
      justify-content: center;
      align-items: center;
      text-align: center;
      margin-bottom: 12px;

      .app-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin: 10px 0;
      }

      .version {
        background-color: #e6f7e6;
        color: #389e3c;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-left: 10px;
      }
    }

    .form-container {
      .form-group {
        margin-bottom: 16px;

        .form-input {
          padding: 10px 12px;
          border: 1px solid #d9d9d9;
          border-radius: 8px;
          font-size: 14px;
          transition: border-color 0.3s ease;

          &:focus {
            border-color: #389e3c;
            outline: none;
          }

          &::placeholder {
            color: #aaa;
          }
        }

        .captcha-container {
          display: flex;
          gap: 12px;

          .captcha-image {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 4px;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            background-color: #f8f9fa;
            cursor: pointer;
            min-height: 44px;
            overflow: hidden;

            &:hover {
              background-color: #e6f7e6;
            }

            .captcha-img {
              width: 100%;
              height: 100%;
            }
          }

          .captcha-text {
            font-size: 18px;
            font-weight: bold;
            color: #333;
          }
        }
      }

      .form-button {
        width: 100%;
        padding: 16px;
        background-color: #389e3c;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        line-height: 24px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin-top: 36px;

        &:active {
          background-color: #2e7d32;
        }

        &:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
      }

      .footer-info {
        margin-top: 16px;
        margin-bottom: 12px;
        text-align: center;

        .info-text {
          font-size: 14px;
          color: #666;
          margin: 4px 0;
        }
      }
    }
  }

  .login-footer {
    padding-bottom: 36px;
    text-align: center;
    color: #666;
    font-size: 12px;

    .login-footer-text {
      margin-top: 6px;

      text {
        margin: 4px 0;
      }
    }
  }
}
</style>
