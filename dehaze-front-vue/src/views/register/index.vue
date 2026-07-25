<template>
  <div class="login-container">
    <el-card class="!border-none !bg-transparent !rounded-4% w-100 <sm:w-85">
      <div class="text-center relative">
        <h2>{{ defaultSettings.title }}</h2>
        <el-tag class="ml-2 absolute-rt">{{ defaultSettings.version }}</el-tag>
      </div>

      <el-form
        ref="registerFormRef"
        :model="registerData"
        :rules="registerRules"
        class="login-form"
      >
        <el-form-item prop="username">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="user" />
            <el-input
              v-model="registerData.username"
              placeholder="用户名（3-32位字母、数字、下划线）"
              class="h-[48px]"
              name="username"
              size="large"
            />
          </div>
        </el-form-item>

        <el-form-item prop="nickname">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="user" />
            <el-input
              v-model="registerData.nickname"
              placeholder="昵称"
              class="h-[48px]"
              name="nickname"
              size="large"
            />
          </div>
        </el-form-item>

        <el-form-item prop="password">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="lock" />
            <el-input
              v-model="registerData.password"
              placeholder="密码（6-20位，含字母和数字）"
              class="h-[48px] pr-2"
              name="password"
              show-password
              size="large"
              type="password"
            />
          </div>
        </el-form-item>

        <el-form-item prop="confirmPassword">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="lock" />
            <el-input
              v-model="registerData.confirmPassword"
              placeholder="确认密码"
              class="h-[48px] pr-2"
              show-password
              size="large"
              type="password"
              @keyup.enter="handleRegister"
            />
          </div>
        </el-form-item>

        <el-form-item prop="captchaCode">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="captcha" />
            <el-input
              v-model="registerData.captchaCode"
              placeholder="验证码"
              auto-complete="off"
              class="flex-1"
              size="large"
              @keyup.enter="handleRegister"
            />
            <el-image
              :src="captchaBase64"
              class="rounded-tr-md rounded-br-md cursor-pointer h-[48px]"
              @click="getCaptcha"
            />
          </div>
        </el-form-item>

        <el-button
          :loading="loading"
          class="w-full"
          size="large"
          type="primary"
          @click.prevent="handleRegister"
        >
          注 册
        </el-button>
      </el-form>

      <div class="text-center mt-4">
        <router-link to="/login">已有账号？立即登录</router-link>
      </div>
    </el-card>
  </div>
</template>

<script lang="ts" setup>
import router from "@/router";
import defaultSettings from "@/settings";
import { AuthAPI, RegisterData } from "dehaze-sdk-js";

const loading = ref(false);
const captchaBase64 = ref("");
const registerFormRef = ref(ElForm);

const registerData = ref<RegisterData & { confirmPassword: string }>({
  username: "",
  password: "",
  nickname: "",
  confirmPassword: "",
  captchaCode: "",
});

const registerRules = {
  username: [{ required: true, trigger: "blur", message: "请输入用户名" }],
  nickname: [{ required: true, trigger: "blur", message: "请输入昵称" }],
  password: [{ required: true, trigger: "blur", message: "请输入密码" }],
  confirmPassword: [{ required: true, trigger: "blur", message: "请确认密码" }],
  captchaCode: [{ required: true, trigger: "blur", message: "请输入验证码" }],
};

function getCaptcha() {
  AuthAPI.getCaptcha().then((data) => {
    registerData.value.captchaKey = data.captchaKey;
    captchaBase64.value = data.captchaBase64;
  });
}

function handleRegister() {
  registerFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    if (registerData.value.password !== registerData.value.confirmPassword) {
      ElMessage.error("两次密码输入不一致");
      return;
    }
    loading.value = true;
    AuthAPI.register({
      username: registerData.value.username,
      password: registerData.value.password,
      nickname: registerData.value.nickname,
      captchaKey: registerData.value.captchaKey,
      captchaCode: registerData.value.captchaCode,
    })
      .then(() => {
        ElMessage.success("注册成功，请登录");
        router.push("/login");
      })
      .catch(() => {
        registerData.value.captchaCode = "";
        getCaptcha();
      })
      .finally(() => {
        loading.value = false;
      });
  });
}

onMounted(() => {
  getCaptcha();
});
</script>

<style lang="scss" scoped>
html.dark .login-container {
  background: url("@/assets/images/login-bg-dark.jpg") no-repeat center right;
}

.login-container {
  overflow-y: auto;
  padding-top: var(--titlebar-h, 0px);
  background: url("@/assets/images/login-bg.jpg") no-repeat center right;
  @apply wh-full flex-center;

  .login-form {
    padding: 30px 10px;
  }
}

.el-form-item {
  background: var(--el-input-bg-color);
  border: 1px solid var(--el-border-color);
  border-radius: 5px;
}
</style>
