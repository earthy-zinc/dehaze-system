<template>
  <PageLayout level="L2" title="套餐管理">
    <view class="page-body">
      <view class="list-row list-row-head">
        <text class="cell">名称</text>
        <text class="cell">售价</text>
        <text class="cell">权益</text>
        <text class="cell">状态</text>
        <text class="cell">操作</text>
      </view>
      <view v-for="item in list" :key="item.id" class="list-row">
        <text class="cell">{{ item.name }}</text>
        <text class="cell">¥{{ fenToYuan(item.salePrice) }}</text>
        <text class="cell">
          {{
            item.packageType === "credit"
              ? `${item.creditAmount ?? 0} 积分`
              : `${item.periodDays}天`
          }}
        </text>
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
            v-model="form.salePrice"
            type="digit"
            placeholder="售价"
          />
        </view>
        <view class="form-row">
          <text class="form-label">原价</text>
          <input
            class="form-input"
            v-model="form.originalPrice"
            type="digit"
            placeholder="原价"
          />
        </view>
        <view class="form-row">
          <text class="form-label">类型</text>
          <picker
            mode="selector"
            :range="packageTypeLabels"
            :value="packageTypeIndex"
            :disabled="!!form.id"
            @change="handlePackageTypeChange"
          >
            <text class="form-input">{{
              packageTypeLabels[packageTypeIndex]
            }}</text>
          </picker>
        </view>
        <view v-if="form.packageType === 'credit'" class="form-row">
          <text class="form-label">积分数量</text>
          <input
            class="form-input"
            v-model.number="form.creditAmount"
            type="number"
            placeholder="到账积分"
          />
        </view>
        <template v-if="form.packageType === 'vip'">
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
        </template>
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
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import FabButton from "@/components/common/FabButton.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { PackageAPI } from "dehaze-sdk-js";
import type { PackageForm } from "dehaze-sdk-js";
import { fenToYuan, yuanToFen } from "@/utils/format";

const list = ref<any[]>([]);
const showForm = ref(false);
const form = ref<any>({
  name: "",
  salePrice: "",
  originalPrice: "",
  levelCode: "level_1",
  period: "monthly",
  periodDays: 30,
  description: "",
  status: 1,
});
const saving = ref(false);

/** 商品类型决定必填项：积分卡要 creditAmount，会员卡要 levelCode/period/periodDays */
const packageTypeValues = ["credit", "vip"] as const;
const packageTypeLabels = ["积分卡", "会员卡"];
const packageTypeIndex = computed(() =>
  Math.max(0, packageTypeValues.indexOf(form.value.packageType))
);
const handlePackageTypeChange = (e: any) => {
  form.value.packageType = packageTypeValues[Number(e.detail.value)];
};

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
      packageType: item.packageType,
      creditAmount: item.creditAmount ?? 0,
      // 接口金额单位为分，表单以元回填
      salePrice: fenToYuan(item.salePrice),
      originalPrice: fenToYuan(item.originalPrice),
      levelCode: item.levelCode,
      period: item.period,
      periodDays: item.periodDays,
      description: item.description || "",
      status: item.status,
    };
  } else {
    form.value = {
      name: "",
      packageType: "credit",
      creditAmount: 0,
      salePrice: "",
      originalPrice: "",
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
    const data: PackageForm =
      form.value.packageType === "credit"
        ? {
            name: form.value.name,
            packageType: "credit",
            creditAmount: Number(form.value.creditAmount),
            salePrice: yuanToFen(form.value.salePrice),
            originalPrice: yuanToFen(form.value.originalPrice),
            description: form.value.description,
            status: form.value.status,
          }
        : {
            name: form.value.name,
            packageType: "vip",
            levelCode: form.value.levelCode,
            period: form.value.period,
            periodDays: Number(form.value.periodDays),
            salePrice: yuanToFen(form.value.salePrice),
            originalPrice: yuanToFen(form.value.originalPrice),
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
  font-size: 26rpx;
  border-bottom: 1rpx solid $color-border;

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
  font-weight: 600;
  color: $color-text-secondary;
  background: $color-bg-secondary;
}

.tag {
  padding: 4rpx 12rpx;
  font-size: $font-xs;
  border-radius: $radius-sm;
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
  font-size: $font-sm;
  line-height: 1.6;
  border-radius: $radius-sm;

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
  width: 90vw;
  max-height: 80vh;
  padding: 30rpx;
  overflow-y: auto;
}

.popup-title {
  margin-bottom: 20rpx;
  font-size: 32rpx;
  font-weight: bold;
}

.form-row {
  display: flex;
  align-items: center;
  padding: 20rpx 0;
  border-bottom: 1rpx solid $color-border;

  .form-label {
    flex-shrink: 0;
    width: 120rpx;
    font-size: 28rpx;
    color: $color-text-secondary;
  }

  .form-input {
    flex: 1;
    font-size: 28rpx;
  }
}
</style>
