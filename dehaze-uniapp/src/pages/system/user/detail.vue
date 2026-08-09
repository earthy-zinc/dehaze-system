<template>
  <PageLayout level="L2" :title="isEdit ? '编辑用户' : '新增用户'">
    <view class="page-body">
      <view class="form-row">
        <text class="form-label"><text class="required">*</text>用户名</text>
        <input
          class="form-input"
          v-model="form.username"
          placeholder="请输入用户名"
        />
      </view>
      <view class="form-row">
        <text class="form-label"><text class="required">*</text>昵称</text>
        <input
          class="form-input"
          v-model="form.nickname"
          placeholder="请输入昵称"
        />
      </view>
      <view v-if="!isEdit" class="form-row">
        <text class="form-label"><text class="required">*</text>密码</text>
        <input
          class="form-input"
          v-model="form.password"
          password
          placeholder="请输入密码"
        />
      </view>
      <view class="form-row">
        <text class="form-label">手机号</text>
        <input
          class="form-input"
          v-model="form.mobile"
          placeholder="请输入手机号"
        />
      </view>
      <view class="form-row">
        <text class="form-label">邮箱</text>
        <input
          class="form-input"
          v-model="form.email"
          placeholder="请输入邮箱"
        />
      </view>
      <view class="form-row">
        <text class="form-label">角色</text>
        <view class="role-picker" @click="showRolePicker = true">
          <text v-if="selectedRoles.length">{{
            selectedRoles.map((r) => r.label).join("、")
          }}</text>
          <text v-else class="placeholder">请选择角色</text>
          <SvgIcon name="arrow-right" />
        </view>
      </view>
      <view v-if="isEdit" class="form-row">
        <text class="form-label">状态</text>
        <switch
          :checked="form.status === 1"
          @change="(e: any) => (form.status = e.detail.value ? 1 : 0)"
        />
      </view>
      <view class="btn-area">
        <button
          v-if="isEdit ? canEdit : canAdd"
          class="btn btn-primary"
          :loading="submitting"
          @click="handleSubmit"
        >
          保存
        </button>
        <button
          v-if="isEdit && canEdit"
          class="btn btn-danger"
          :loading="resetting"
          @click="handleResetPwd"
        >
          重置密码
        </button>
      </view>
    </view>
    <Popup
      :show="showRolePicker"
      mode="center"
      round
      @close="showRolePicker = false"
    >
      <view class="popup-body">
        <scroll-view scroll-y class="role-scroll">
          <view
            v-for="role in allRoles"
            :key="role.value"
            class="role-item"
            @click="toggleRole(role)"
          >
            <checkbox :checked="selectedRoleIds.includes(role.value)" />
            <text class="role-label">{{ role.label }}</text>
          </view>
        </scroll-view>
        <view class="popup-footer">
          <button class="btn btn-primary" @click="showRolePicker = false">
            确定
          </button>
        </view>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import Popup from "@/components/common/Popup.vue";
import { UserAPI, RoleAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:user:add"));
const canEdit = computed(() => authStore.hasPerm("sys:user:edit"));

const id = ref(0);
const isEdit = computed(() => id.value > 0);
const form = ref<any>({
  username: "",
  nickname: "",
  password: "",
  mobile: "",
  email: "",
  status: 1,
});
const selectedRoleIds = ref<number[]>([]);
const allRoles = ref<any[]>([]);
const showRolePicker = ref(false);
const submitting = ref(false);
const resetting = ref(false);

const selectedRoles = computed(() =>
  allRoles.value.filter((r) => selectedRoleIds.value.includes(r.value))
);

onLoad((options: any) => {
  id.value = +(options?.id || 0);
  fetchRoles();
  if (isEdit.value) fetchUser();
});

const fetchRoles = async () => {
  try {
    allRoles.value = (await RoleAPI.getOptions()) || [];
  } catch {}
};
const fetchUser = async () => {
  try {
    const d = await UserAPI.getFormData(id.value);
    form.value = {
      username: d.username,
      nickname: d.nickname,
      password: "",
      mobile: d.mobile || "",
      email: d.email || "",
      status: d.status,
    };
    selectedRoleIds.value = d.roleIds || [];
  } catch {}
};
const toggleRole = (role: any) => {
  const roleValue = role.value ?? role.id;
  const idx = selectedRoleIds.value.indexOf(roleValue);
  if (idx > -1) selectedRoleIds.value.splice(idx, 1);
  else selectedRoleIds.value.push(roleValue);
};
const handleSubmit = async () => {
  submitting.value = true;
  try {
    const data: any = { ...form.value, roleIds: selectedRoleIds.value };
    if (isEdit.value) {
      delete data.password;
      await UserAPI.update(id.value, data);
    } else {
      if (!data.password?.trim()) {
        uni.showToast({ title: "请输入密码", icon: "none" });
        submitting.value = false;
        return;
      }
      await UserAPI.add(data);
    }
    uni.showToast({ title: "保存成功", icon: "success" });
    setTimeout(() => uni.navigateBack(), 500);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    submitting.value = false;
  }
};
const handleResetPwd = async () => {
  resetting.value = true;
  try {
    await UserAPI.updatePassword(id.value, "reset123456");
    uni.showToast({ title: "密码已重置", icon: "success" });
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  } finally {
    resetting.value = false;
  }
};
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.form-row {
  display: flex;
  align-items: center;
  padding: 20rpx 0;
  border-bottom: 1rpx solid $color-border;
}
.form-label {
  width: 180rpx;
  flex-shrink: 0;
  color: $color-text-primary;
}
.required {
  color: $color-danger;
  margin-right: 4rpx;
}
.form-input {
  flex: 1;
  font-size: $font-sm;
}
.role-picker {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.placeholder {
  color: $color-text-placeholder;
}
.btn-area {
  display: flex;
  gap: 20rpx;
  margin-top: 40rpx;
}
.btn {
  padding: 8rpx 20rpx;
  border-radius: $radius-sm;
  font-size: $font-sm;

  &::after {
    border: none;
  }
}
.btn-primary {
  background: $color-primary;
  color: $color-white;
}
.btn-danger {
  background: $color-danger;
  color: $color-white;
}
.popup-body {
  padding: 30rpx;
  width: 86vw;
}
.role-scroll {
  max-height: 400rpx;
}
.role-item {
  display: flex;
  align-items: center;
  gap: 16rpx;
  padding: 16rpx 0;
  border-bottom: 1rpx solid $color-border;
}
.role-label {
  flex: 1;
}
.popup-footer {
  padding-top: 20rpx;
}
</style>
