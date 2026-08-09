<template>
  <PageLayout level="L2" title="字典管理">
    <view class="page-body">
      <view class="list-row" v-for="dict in list" :key="dict.id">
        <text class="cell" @click="goItems(dict)">{{ dict.name }}</text>
        <text class="cell" @click="goItems(dict)">{{ dict.code }}</text>
        <view class="cell" @click="goItems(dict)">
          <view
            class="tag"
            :class="dict.status === 1 ? 'tag-success' : 'tag-danger'"
          >
            {{ dict.status === 1 ? "启用" : "禁用" }}
          </view>
        </view>
        <view class="cell row-actions">
          <SvgIcon v-if="canEdit" name="edit-pen" @click="editType(dict)" />
          <SvgIcon name="arrow-right" @click="goItems(dict)" />
        </view>
      </view>
      <view v-if="list.length === 0" class="empty-tip">暂无字典类型</view>
    </view>
    <FabButton v-if="canAdd" @click="openAdd">
      <SvgIcon name="plus" color="#fff" size="24" />
    </FabButton>
    <Popup :show="showForm" mode="center" round @close="showForm = false">
      <view class="popup-body">
        <view class="popup-title">{{
          editId ? "编辑字典类型" : "新增字典类型"
        }}</view>
        <view class="form-row">
          <text class="form-label">名称</text>
          <input
            class="form-input"
            v-model="form.name"
            placeholder="字典名称"
          />
        </view>
        <view class="form-row">
          <text class="form-label">编码</text>
          <input
            class="form-input"
            v-model="form.code"
            placeholder="字典编码"
          />
        </view>
        <view class="form-row">
          <text class="form-label">状态</text>
          <switch
            :checked="form.status === 1"
            @change="(e: any) => (form.status = e.detail.value ? 1 : 0)"
          />
        </view>
        <button class="btn btn-primary" :loading="saving" @click="handleSave">
          保存
        </button>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import FabButton from "@/components/common/FabButton.vue";
import Popup from "@/components/common/Popup.vue";
import { DictAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:dict:type:add"));
const canEdit = computed(() => authStore.hasPerm("sys:dict:type:edit"));

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
const openAdd = () => {
  editId.value = 0;
  form.value = { name: "", code: "", status: 1 };
  showForm.value = true;
};
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
.list-row {
  display: flex;
  align-items: center;
  padding: 24rpx 20rpx;
  border-bottom: 1rpx solid $color-border;

  .cell {
    flex: 1;
    font-size: $font-sm;
    color: $color-text-primary;
  }

  .row-actions {
    display: flex;
    gap: 16rpx;
  }
}
.tag {
  display: inline-block;
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-success {
  background: $color-success-bg;
  color: $color-success;
}
.tag-danger {
  background: $color-danger-bg;
  color: $color-danger;
}
.popup-body {
  padding: 30rpx;
  width: 90vw;
}
.popup-title {
  font-size: $font-lg;
  font-weight: bold;
  margin-bottom: 20rpx;
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
.form-input {
  flex: 1;
  font-size: $font-sm;
}
.btn {
  width: 100%;
  margin-top: 40rpx;
  padding: 16rpx 0;
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
</style>
