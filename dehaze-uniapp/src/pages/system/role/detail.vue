<template>
  <PageLayout level="L2" :title="isEdit ? '编辑角色' : '新增角色'">
    <view class="page-body">
      <u-form :model="form">
        <u-form-item label="角色名称" required
          ><u-input v-model="form.name" placeholder="请输入角色名称"
        /></u-form-item>
        <u-form-item label="角色编码" required
          ><u-input v-model="form.code" placeholder="请输入角色编码"
        /></u-form-item>
        <u-form-item label="排序"
          ><u-input
            v-model.number="form.sort"
            type="number"
            placeholder="请输入排序"
        /></u-form-item>
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
        <u-button v-if="isEdit" type="warning" @click="goPermission"
          >权限分配</u-button
        >
      </view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import { RoleAPI } from "dehaze-sdk-js";

const id = ref(0);
const isEdit = computed(() => id.value > 0);
const form = ref<any>({ name: "", code: "", sort: 0, status: 1 });
const submitting = ref(false);

onLoad((options: any) => {
  id.value = +(options?.id || 0);
  if (isEdit.value) fetchRole();
});

const fetchRole = async () => {
  try {
    const d = await RoleAPI.getFormData(id.value);
    form.value = {
      name: d.name,
      code: d.code,
      sort: d.sort || 0,
      status: d.status,
    };
  } catch {}
};
const handleSubmit = async () => {
  submitting.value = true;
  try {
    if (isEdit.value) await RoleAPI.update(id.value, form.value);
    else await RoleAPI.add(form.value);
    uni.showToast({ title: "保存成功", icon: "success" });
    setTimeout(() => uni.navigateBack(), 500);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    submitting.value = false;
  }
};
const goPermission = () =>
  uni.navigateTo({ url: `/pages/system/role/permission?id=${id.value}` });
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.btn-area {
  display: flex;
  gap: 20rpx;
  margin-top: 40rpx;
}
</style>
