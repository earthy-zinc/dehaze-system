<template>
  <PageLayout level="L2" :title="isEdit ? '编辑用户' : '新增用户'">
    <view class="page-body">
      <u-form :model="form" ref="formRef">
        <u-form-item label="用户名" required
          ><u-input v-model="form.username" placeholder="请输入用户名"
        /></u-form-item>
        <u-form-item label="昵称" required
          ><u-input v-model="form.nickname" placeholder="请输入昵称"
        /></u-form-item>
        <u-form-item v-if="!isEdit" label="密码" required
          ><u-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
        /></u-form-item>
        <u-form-item label="手机号"
          ><u-input v-model="form.mobile" placeholder="请输入手机号"
        /></u-form-item>
        <u-form-item label="邮箱"
          ><u-input v-model="form.email" placeholder="请输入邮箱"
        /></u-form-item>
        <u-form-item label="角色">
          <view class="role-picker" @click="showRolePicker = true">
            <text v-if="selectedRoles.length">{{
              selectedRoles.map((r) => r.label).join("、")
            }}</text>
            <text v-else class="placeholder">请选择角色</text>
            <SvgIcon name="arrow-right" />
          </view>
        </u-form-item>
        <u-form-item label="状态" v-if="isEdit">
          <u-switch
            :checked="form.status === 1"
            @change="(val: boolean) => (form.status = val ? 1 : 0)"
          />
        </u-form-item>
      </u-form>
      <view class="btn-area">
        <u-button type="primary" @click="handleSubmit" :loading="submitting"
          >保存</u-button
        >
        <u-button
          v-if="isEdit"
          type="error"
          @click="handleResetPwd"
          :loading="resetting"
          >重置密码</u-button
        >
      </view>
    </view>
    <u-popup :show="showRolePicker" @close="showRolePicker = false" round>
      <view class="popup-content">
        <scroll-view scroll-y class="role-scroll">
          <view
            v-for="role in allRoles"
            :key="role.value"
            class="role-item"
            @click="toggleRole(role)"
          >
            <u-checkbox :checked="selectedRoleIds.includes(role.value)" />
            <text>{{ role.label }}</text>
          </view>
        </scroll-view>
        <view class="popup-footer">
          <u-button type="primary" @click="showRolePicker = false"
            >确定</u-button
          >
        </view>
      </view>
    </u-popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { UserAPI, RoleAPI } from "dehaze-sdk-js";

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
.role-picker {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.placeholder {
  color: $color-text-secondary;
}
.btn-area {
  display: flex;
  gap: 20rpx;
  margin-top: 40rpx;
}
.popup-content {
  padding: 20rpx;
  max-height: 600rpx;
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
.popup-footer {
  padding-top: 20rpx;
}
</style>
