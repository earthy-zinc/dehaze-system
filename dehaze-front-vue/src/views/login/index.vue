<template>
  <div class="login-container">
    <!-- 顶部 -->
    <div
      class="absolute-lt flex-x-end p-3 w-full"
      style="top: var(--titlebar-h, 0)"
    >
      <el-switch
        v-model="isDark"
        :active-icon="Moon"
        :inactive-icon="Sunny"
        inline-prompt
        @change="toggleTheme"
      />
      <lang-select class="ml-2 cursor-pointer" />
    </div>
    <!-- 登录表单 -->
    <el-card class="!border-none !bg-transparent !rounded-4% w-100 <sm:w-85">
      <div class="text-center relative">
        <h2>{{ defaultSettings.title }}</h2>
        <el-tag class="ml-2 absolute-rt">{{ defaultSettings.version }}</el-tag>
      </div>

      <el-form
        ref="loginFormRef"
        :model="loginData"
        :rules="loginRules"
        class="login-form"
      >
        <!-- 用户名 -->
        <el-form-item prop="username">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="user" />
            <el-input
              ref="username"
              v-model="loginData.username"
              :placeholder="$t('login.username')"
              class="h-[48px]"
              name="username"
              size="large"
            />
          </div>
        </el-form-item>

        <!-- 密码 -->
        <el-tooltip
          :content="$t('login.capsLock')"
          :visible="isCapslock"
          placement="right"
        >
          <el-form-item prop="password">
            <div class="flex-y-center w-full">
              <svg-icon class="mx-2" icon-class="lock" />
              <el-input
                v-model="loginData.password"
                :placeholder="$t('login.password')"
                class="h-[48px] pr-2"
                name="password"
                show-password
                size="large"
                type="password"
                @keyup="checkCapslock"
                @keyup.enter="handleLogin"
              />
            </div>
          </el-form-item>
        </el-tooltip>

        <!-- 验证码 -->
        <el-form-item prop="captchaCode">
          <div class="flex-y-center w-full">
            <svg-icon class="mx-2" icon-class="captcha" />
            <el-input
              v-model="loginData.captchaCode"
              :placeholder="$t('login.captchaCode')"
              auto-complete="off"
              class="flex-1"
              size="large"
              @keyup.enter="handleLogin"
            />

            <el-image
              :src="captchaBase64"
              class="rounded-tr-md rounded-br-md cursor-pointer h-[48px]"
              @click="getCaptcha"
            />
          </div>
        </el-form-item>

        <!-- 记住我 -->
        <div class="flex-x-between w-full mb-4">
          <el-checkbox v-model="loginData.rememberMe">
            记住我（7天内免登录）
          </el-checkbox>
        </div>

        <!-- 登录按钮 -->
        <el-button
          :loading="loading"
          class="w-full"
          size="large"
          type="primary"
          @click.prevent="handleLogin"
          >{{ $t("login.login") }}
        </el-button>

        <!-- 账号密码提示 -->
        <div class="mt-10 text-sm">
          <span>{{ $t("login.username") }}: admin</span>
          <span class="ml-4"> {{ $t("login.password") }}: 123456</span>
        </div>
      </el-form>
    </el-card>

    <!-- ICP备案 -->
    <div v-show="icpVisible" class="absolute bottom-1 text-[10px] text-center">
      <p>
        Copyright © 2022 - 2024 Peixin Wu All Rights Reserved. 武沛鑫 版权所有
      </p>
      <p>渝ICP备2024111923号-2</p>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ThemeEnum } from "@/enums/ThemeEnum";
import router from "@/router";
import defaultSettings from "@/settings";
import { useSettingsStore, useUserStore } from "@/store";
import { Moon, Sunny } from "@element-plus/icons-vue";

import { AuthAPI, LoginData } from "dehaze-sdk-js";
import { LocationQuery, LocationQueryValue, useRoute } from "vue-router";

const userStore = useUserStore();
const settingsStore = useSettingsStore();

const { t } = useI18n();

const isDark = ref(settingsStore.theme === ThemeEnum.DARK);
const icpVisible = ref(true);
const loading = ref(false);
const isCapslock = ref(false);
const captchaBase64 = ref();
const loginFormRef = ref(ElForm);
const { height } = useWindowSize();

const loginData = ref<LoginData>({
  username: "admin",
  password: "123456",
  rememberMe: true,
});

const loginRules = computed(() => {
  return {
    username: [
      {
        required: true,
        trigger: "blur",
        message: t("login.message.username.required"),
      },
    ],
    password: [
      {
        required: true,
        trigger: "blur",
        message: t("login.message.password.required"),
      },
      {
        min: 6,
        message: t("login.message.password.min"),
        trigger: "blur",
      },
    ],
    captchaCode: [
      {
        required: true,
        trigger: "blur",
        message: t("login.message.captchaCode.required"),
      },
    ],
  };
});

function getCaptcha() {
  AuthAPI.getCaptcha().then((data) => {
    loginData.value.captchaKey = data.captchaKey;
    captchaBase64.value = data.captchaBase64;
  });
}

const route = useRoute();

function handleLogin() {
  loginFormRef.value.validate((valid: boolean) => {
    if (valid) {
      loading.value = true;
      userStore
        .login(loginData.value)
        .then(() => {
          const query: LocationQuery = route.query;
          const redirect = (query.redirect as LocationQueryValue) ?? "/";
          const otherQueryParams = Object.keys(query).reduce(
            (acc: any, cur: string) => {
              if (cur !== "redirect") {
                acc[cur] = query[cur];
              }
              return acc;
            },
            {}
          );

          router.push({ path: redirect, query: otherQueryParams });
        })
        .catch(() => {
          getCaptcha();
        })
        .finally(() => {
          loading.value = false;
        });
    }
  });
}

const toggleTheme = () => {
  const newTheme =
    settingsStore.theme === ThemeEnum.DARK ? ThemeEnum.LIGHT : ThemeEnum.DARK;
  settingsStore.changeTheme(newTheme);
};

watchEffect(() => {
  if (height.value < 600) {
    icpVisible.value = false;
  } else {
    icpVisible.value = true;
  }
});

function checkCapslock(event: KeyboardEvent) {
  if (event instanceof KeyboardEvent) {
    isCapslock.value = event.getModifierState("CapsLock");
  }
}

onMounted(() => {
  getCaptcha();
  nextTick(() => {
    loginFormRef.value?.fields?.[0]?.ref?.focus?.();
  });
});
</script>

<style lang="scss" scoped>
html.dark .login-container {
  background: url("@/assets/images/login-bg-dark.jpg") no-repeat center right;
}

.login-container {
  padding-top: var(--titlebar-h, 0);
  overflow-y: auto;
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

:deep(.el-input) {
  .el-input__wrapper {
    padding: 0;
    background-color: transparent;
    box-shadow: none;

    &.is-focus,
    &:hover {
      box-shadow: none !important;
    }

    input:-webkit-autofill {
      /* 通过延时渲染背景色变相去除背景颜色 */
      transition: background-color 1000s ease-in-out 0s;
    }
  }
}
</style>
