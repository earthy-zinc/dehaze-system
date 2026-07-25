<template>
  <view class="login-container">
    <view class="login-card">
      <view class="title">用户注册</view>
      <u-form :model="form" ref="formRef">
        <u-form-item prop="username" borderBottom>
          <u-input
            v-model="form.username"
            placeholder="用户名（3-32位字母数字下划线）"
          />
        </u-form-item>
        <u-form-item prop="nickname" borderBottom>
          <u-input v-model="form.nickname" placeholder="昵称" />
        </u-form-item>
        <u-form-item prop="password" borderBottom>
          <u-input
            v-model="form.password"
            type="password"
            placeholder="密码（6-20位含字母和数字）"
          />
        </u-form-item>
        <u-form-item prop="confirmPassword" borderBottom>
          <u-input
            v-model="form.confirmPassword"
            type="password"
            placeholder="确认密码"
          />
        </u-form-item>
        <u-form-item prop="captchaCode" borderBottom>
          <u-input v-model="form.captchaCode" placeholder="验证码" />
          <template #right>
            <image
              :src="captchaBase64"
              style="width: 100px; height: 40px"
              @click="getCaptcha"
            />
          </template>
        </u-form-item>
      </u-form>
      <u-button type="primary" :loading="loading" @click="handleRegister" block
        >注 册</u-button
      >
      <view class="link" @click="goLogin">已有账号？立即登录</view>
    </view>
  </view>
</template>

<script setup>
import { ref } from "vue";
import { AuthAPI } from "dehaze-sdk-js";

const form = ref({
  username: "",
  nickname: "",
  password: "",
  confirmPassword: "",
  captchaCode: "",
  captchaKey: "",
});
const captchaBase64 = ref("");
const loading = ref(false);

function getCaptcha() {
  AuthAPI.getCaptcha().then((d) => {
    form.value.captchaKey = d.captchaKey;
    captchaBase64.value = d.captchaBase64;
  });
}

function handleRegister() {
  if (!form.value.username) {
    uni.showToast({ title: "请输入用户名", icon: "none" });
    return;
  }
  if (!form.value.nickname) {
    uni.showToast({ title: "请输入昵称", icon: "none" });
    return;
  }
  if (!form.value.password) {
    uni.showToast({ title: "请输入密码", icon: "none" });
    return;
  }
  if (form.value.password !== form.value.confirmPassword) {
    uni.showToast({ title: "两次密码不一致", icon: "none" });
    return;
  }
  if (!form.value.captchaCode) {
    uni.showToast({ title: "请输入验证码", icon: "none" });
    return;
  }

  loading.value = true;
  AuthAPI.register({
    username: form.value.username,
    password: form.value.password,
    nickname: form.value.nickname,
    captchaKey: form.value.captchaKey,
    captchaCode: form.value.captchaCode,
  })
    .then(() => {
      uni.showToast({ title: "注册成功", icon: "success" });
      setTimeout(() => uni.reLaunch({ url: "/pages/login/index" }), 1000);
    })
    .catch(() => {
      getCaptcha();
      form.value.captchaCode = "";
    })
    .finally(() => {
      loading.value = false;
    });
}

function goLogin() {
  uni.reLaunch({ url: "/pages/login/index" });
}

onMounted(() => {
  getCaptcha();
});
</script>

<style lang="scss" scoped>
.login-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: #f5f7fa;
  padding: 32px;
}
.login-card {
  width: 100%;
  max-width: 360px;
  background: #fff;
  border-radius: 12px;
  padding: 32px 24px;
}
.title {
  font-size: 24px;
  font-weight: bold;
  text-align: center;
  margin-bottom: 24px;
}
.link {
  text-align: center;
  margin-top: 16px;
  color: #389e3c;
}
</style>
