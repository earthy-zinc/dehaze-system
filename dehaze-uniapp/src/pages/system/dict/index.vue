<template>
  <PageLayout level="L2" title="字典管理">
    <view class="page-body">
      <u-table>
        <u-tr v-for="dict in list" :key="dict.id">
          <u-td @click="goItems(dict)">{{ dict.name }}</u-td>
          <u-td @click="goItems(dict)">{{ dict.code }}</u-td>
          <u-td @click="goItems(dict)">
            <u-tag
              :text="dict.status === 1 ? '启用' : '禁用'"
              :type="dict.status === 1 ? 'success' : 'error'"
              size="mini"
            />
          </u-td>
          <u-td>
            <view class="row-actions">
              <SvgIcon name="edit-pen" @click="editType(dict)" />
              <SvgIcon name="arrow-right" @click="goItems(dict)" />
            </view>
          </u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="list.length === 0" text="暂无字典类型" />
    </view>
    <view class="fab-btn" @click="showForm = true"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></view>
    <u-popup :show="showForm" @close="showForm = false" round>
      <view class="popup-content">
        <view class="popup-title">{{
          editId ? "编辑字典类型" : "新增字典类型"
        }}</view>
        <u-form :model="form">
          <u-form-item label="名称"
            ><u-input v-model="form.name" placeholder="字典名称"
          /></u-form-item>
          <u-form-item label="编码"
            ><u-input v-model="form.code" placeholder="字典编码"
          /></u-form-item>
          <u-form-item label="状态">
            <u-switch
              :checked="form.status === 1"
              @change="(val: boolean) => (form.status = val ? 1 : 0)"
            />
          </u-form-item>
        </u-form>
        <u-button type="primary" @click="handleSave" :loading="saving"
          >保存</u-button
        >
      </view>
    </u-popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { DictAPI } from "dehaze-sdk-js";

const list = ref<any[]>([]);
const showForm = ref(false);
const editId = ref(0);
const form = ref<any>({ name: "", code: "", status: 1 });
const saving = ref(false);

const fetchList = async () => {
  try {
    const res = await DictAPI.getDictTypePage({ pageNum: 1, pageSize: 100 });
    list.value = res.list || [];
  } catch {}
};
const goItems = (dict: any) =>
  uni.navigateTo({
    url: `/pages/system/dict/items?typeCode=${dict.code}&name=${encodeURIComponent(dict.name)}`,
  });
const editType = (dict: any) => {
  editId.value = dict.id;
  form.value = {
    name: dict.name || "",
    code: dict.code || "",
    status: dict.status ?? 1,
  };
  showForm.value = true;
};
const handleSave = async () => {
  saving.value = true;
  try {
    if (editId.value)
      await DictAPI.updateDictType(editId.value, {
        ...form.value,
        id: editId.value,
      });
    else await DictAPI.addDictType(form.value);
    showForm.value = false;
    editId.value = 0;
    fetchList();
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};

fetchList();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.row-actions {
  display: flex;
  gap: 16rpx;
}
.fab-btn {
  position: fixed;
  right: 40rpx;
  bottom: 100rpx;
  width: 96rpx;
  height: 96rpx;
  border-radius: 50%;
  background: $color-primary;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.2);
  z-index: 99;
}
.popup-content {
  padding: 30rpx;
  width: 90vw;
}
.popup-title {
  font-size: 32rpx;
  font-weight: bold;
  margin-bottom: 20rpx;
}
</style>
