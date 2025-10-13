<template>
  <view class="login-container">
    <view class="login-header"></view>

    <view class="login-card">
      <view class="login-header">
        <text class="app-title">图像去雾系统</text>
        <text class="version">1.10.1</text>
      </view>

      <Form class="form-container">
        <view class="form-group">
          <input
            class="form-input"
            placeholder="请输入用户名"
            v-model="formData.username"
          />
        </view>

        <view class="form-group">
          <input
            class="form-input"
            placeholder="请输入密码"
            password
            v-model="formData.password"
          />
        </view>

        <view class="form-group">
          <view class="captcha-container">
            <input
              :class="{ 'form-input': true, error: captchaError }"
              placeholder="请输入验证码"
              v-model="formData.captcha"
            />
            <view class="captcha-image" @click="refreshCaptcha">
              <text class="captcha-text">{{ captchaValue }}</text>
            </view>
          </view>
          <text v-if="captchaError" class="error-message">
            {{ captchaError }}
          </text>
        </view>

        <button class="form-button" @click="handleSubmit" :disabled="loading">
          {{ loading ? "登录中..." : "登 录" }}
        </button>

        <view class="footer-info">
          <text class="info-text">用户名: admin</text>
          <text class="info-text">密码: 123456</text>
        </view>
      </Form>
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

<script setup lang="ts">
import { reactive, ref } from "vue";

const loading = ref(false);
const formData = reactive({
  username: "",
  password: "",
  captcha: "",
});

const captchaValue = ref("");
const captchaError = ref("");
const handleSubmit = () => {};

const refreshCaptcha = () => {};
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
            padding: 8px 12px;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            background-color: #f8f9fa;
            cursor: pointer;
            text-align: center;

            &:hover {
              background-color: #e6f7e6;
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
