<template>
  <PageLayout level="L2" title="套餐管理">
    <view class="page-body">
      <u-table>
        <u-tr v-for="item in list" :key="item.id">
          <u-td>{{ item.name }}</u-td>
          <u-td>¥{{ item.salePrice }}</u-td>
          <u-td>{{ item.periodDays }}天</u-td>
          <u-td>
            <u-tag
              :text="item.status === 1 ? '上架' : '下架'"
              :type="item.status === 1 ? 'success' : 'error'"
              size="mini"
            />
          </u-td>
          <u-td>
            <SvgIcon name="edit-pen" @click="editItem(item)" />
            <u-button
              size="mini"
              :type="item.status === 1 ? 'warning' : 'success'"
              @click="toggleStatus(item)"
            >
              {{ item.status === 1 ? "下架" : "上架" }}
            </u-button>
          </u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="list.length === 0" text="暂无套餐" />
    </view>
    <view class="fab-btn" @click="editItem(null)"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></view>
    <u-popup :show="showForm" @close="showForm = false" round>
      <view class="popup-content">
        <view class="popup-title">{{ form.id ? "编辑套餐" : "新增套餐" }}</view>
        <u-form :model="form">
          <u-form-item label="名称"
            ><u-input v-model="form.name" placeholder="套餐名称"
          /></u-form-item>
          <u-form-item label="售价"
            ><u-input
              v-model.number="form.salePrice"
              type="number"
              placeholder="售价"
          /></u-form-item>
          <u-form-item label="原价"
            ><u-input
              v-model.number="form.originalPrice"
              type="number"
              placeholder="原价"
          /></u-form-item>
          <u-form-item label="等级"
            ><u-input v-model="form.levelCode" placeholder="level_1/level_2/level_3"
          /></u-form-item>
          <u-form-item label="周期"
            ><u-input v-model="form.period" placeholder="monthly/quarterly/yearly"
          /></u-form-item>
          <u-form-item label="天数"
            ><u-input
              v-model.number="form.periodDays"
              type="number"
              placeholder="有效天数"
          /></u-form-item>
          <u-form-item label="描述"
            ><u-input v-model="form.description" placeholder="描述"
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
