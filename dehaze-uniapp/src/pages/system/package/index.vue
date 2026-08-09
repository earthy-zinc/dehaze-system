<template>
  <PageLayout level="L2" title="套餐管理">
    <view class="page-body">
      <view class="list-row list-row-head">
        <text class="cell">名称</text>
        <text class="cell">售价</text>
        <text class="cell">周期</text>
        <text class="cell">状态</text>
        <text class="cell">操作</text>
      </view>
      <view v-for="item in list" :key="item.id" class="list-row">
        <text class="cell">{{ item.name }}</text>
        <text class="cell">¥{{ item.salePrice }}</text>
        <text class="cell">{{ item.periodDays }}天</text>
        <view class="cell">
          <view
            class="tag tag-sm"
            :class="item.status === 1 ? 'tag-success' : 'tag-danger'"
          >
            {{ item.status === 1 ? "上架" : "下架" }}
          </view>
        </view>
        <view class="cell cell-actions">
          <SvgIcon name="edit-pen" @click="editItem(item)" />
          <button
            class="btn btn-sm"
            :class="item.status === 1 ? 'btn-warning' : 'btn-success'"
            @click="toggleStatus(item)"
          >
            {{ item.status === 1 ? "下架" : "上架" }}
          </button>
        </view>
      </view>
      <view v-if="list.length === 0" class="empty-tip">暂无套餐</view>
    </view>
    <FabButton @click="editItem(null)"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></FabButton>
    <Popup :show="showForm" mode="center" round @close="showForm = false">
      <view class="popup-content">
        <view class="popup-title">{{ form.id ? "编辑套餐" : "新增套餐" }}</view>
        <view class="form-row">
          <text class="form-label">名称</text>
          <input
            class="form-input"
            v-model="form.name"
            placeholder="套餐名称"
          />
        </view>
        <view class="form-row">
          <text class="form-label">售价</text>
          <input
            class="form-input"
            v-model.number="form.salePrice"
            type="number"
            placeholder="售价"
          />
        </view>
        <view class="form-row">
          <text class="form-label">原价</text>
          <input
            class="form-input"
            v-model.number="form.originalPrice"
            type="number"
            placeholder="原价"
          />
        </view>
        <view class="form-row">
          <text class="form-label">等级</text>
          <input
            class="form-input"
            v-model="form.levelCode"
            placeholder="level_1/level_2/level_3"
          />
        </view>
        <view class="form-row">
          <text class="form-label">周期</text>
          <input
            class="form-input"
            v-model="form.period"
            placeholder="monthly/quarterly/yearly"
          />
        </view>
        <view class="form-row">
          <text class="form-label">天数</text>
          <input
            class="form-input"
            v-model.number="form.periodDays"
            type="number"
            placeholder="有效天数"
          />
        </view>
        <view class="form-row">
          <text class="form-label">描述</text>
          <input
            class="form-input"
            v-model="form.description"
            placeholder="描述"
          />
        </view>
        <view class="form-row">
          <text class="form-label">状态</text>
          <switch
            :checked="form.status === 1"
            @change="(e: any) => (form.status = e.detail.value ? 1 : 0)"
          />
        </view>
        <button class="btn btn-primary" :disabled="saving" @click="handleSave">
          保存
        </button>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import FabButton from "@/components/common/FabButton.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { PackageAPI } from "dehaze-sdk-js";

const list = ref<any[]>([]);
const showForm = ref(false);
const form = ref<any>({
  name: "",
  salePrice: 0,
  originalPrice: 0,
  levelCode: "level_1",
  period: "monthly",
  periodDays: 30,
  description: "",
  status: 1,
});
const saving = ref(false);

const fetchList = async () => {
  try {
    const res = await PackageAPI.getPage({ pageNum: 1, pageSize: 100 });
    list.value = res.list || [];
  } catch {}
};
const editItem = (item: any) => {
  if (item) {
    form.value = {
      id: item.id,
      name: item.name,
      salePrice: item.salePrice,
      originalPrice: item.originalPrice,
      levelCode: item.levelCode,
      period: item.period,
      periodDays: item.periodDays,
      description: item.description || "",
      status: item.status,
    };
  } else {
    form.value = {
      name: "",
      salePrice: 0,
      originalPrice: 0,
      levelCode: "level_1",
      period: "monthly",
      periodDays: 30,
      description: "",
      status: 1,
    };
  }
  showForm.value = true;
};
const handleSave = async () => {
  saving.value = true;
  try {
    const data = {
      name: form.value.name,
      salePrice: Number(form.value.salePrice),
      originalPrice: Number(form.value.originalPrice),
      levelCode: form.value.levelCode,
      period: form.value.period,
      periodDays: Number(form.value.periodDays),
      description: form.value.description,
      status: form.value.status,
    };
    if (form.value.id) await PackageAPI.update(form.value.id, data);
    else await PackageAPI.add(data);
    showForm.value = false;
    fetchList();
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};
const toggleStatus = async (item: any) => {
  const newStatus = item.status === 1 ? 0 : 1;
  try {
    await PackageAPI.updateStatus(item.id, newStatus);
    item.status = newStatus;
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
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
  padding: 20rpx 16rpx;
  border-bottom: 1rpx solid $color-border;
  font-size: 26rpx;

  .cell {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cell-actions {
    display: flex;
    gap: 12rpx;
    align-items: center;
  }
}
.list-row-head {
  background: $color-bg-secondary;
  font-weight: 600;
  color: $color-text-secondary;
}
.tag {
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-sm {
  padding: 2rpx 10rpx;
}
.tag-success {
  color: $color-success;
  background: $color-success-bg;
}
.tag-danger {
  color: $color-danger;
  background: $color-danger-bg;
}
.btn {
  padding: 8rpx 20rpx;
  border-radius: $radius-sm;
  font-size: $font-sm;
  line-height: 1.6;
  &::after {
    border: none;
  }
}
.btn-sm {
  padding: 4rpx 16rpx;
  font-size: $font-xs;
}
.btn-primary {
  color: $color-white;
  background: $color-primary;
}
.btn-success {
  color: $color-white;
  background: $color-success;
}
.btn-warning {
  color: $color-white;
  background: $color-warning;
}
.popup-content {
  padding: 30rpx;
  width: 90vw;
  max-height: 80vh;
  overflow-y: auto;
}
.popup-title {
  font-size: 32rpx;
  font-weight: bold;
  margin-bottom: 20rpx;
}
.form-row {
  display: flex;
  align-items: center;
  padding: 20rpx 0;
  border-bottom: 1rpx solid $color-border;

  .form-label {
    width: 120rpx;
    flex-shrink: 0;
    font-size: 28rpx;
    color: $color-text-secondary;
  }

  .form-input {
    flex: 1;
    font-size: 28rpx;
  }
}
</style>
