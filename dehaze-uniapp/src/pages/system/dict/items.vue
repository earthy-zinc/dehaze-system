<template>
  <PageLayout level="L2" :title="dictName + ' - 字典项'">
    <view class="page-body">
      <u-table>
        <u-tr v-for="item in list" :key="item.id">
          <u-td>{{ item.name }}</u-td>
          <u-td>{{ item.value }}</u-td>
          <u-td>
            <u-tag
              :text="item.status === 1 ? '启用' : '禁用'"
              :type="item.status === 1 ? 'success' : 'error'"
              size="mini"
            />
          </u-td>
          <u-td>
            <SvgIcon name="edit-pen" @click="editItem(item)" />
            <SvgIcon
              name="trash"
              @click="delItem(item.id)"
              color="$color-error"
            />
          </u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="list.length === 0" text="暂无字典项" />
    </view>
    <view class="fab-btn" @click="editItem(null)"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></view>
    <u-popup :show="showForm" @close="showForm = false" round>
      <view class="popup-content">
        <view class="popup-title">{{
          editId ? "编辑字典项" : "新增字典项"
        }}</view>
        <u-form :model="form">
          <u-form-item label="标签"
            ><u-input v-model="form.name" placeholder="显示标签"
          /></u-form-item>
          <u-form-item label="值"
            ><u-input v-model="form.value" placeholder="字典值"
          /></u-form-item>
          <u-form-item label="排序"
            ><u-input
              v-model.number="form.sort"
              type="number"
              placeholder="排序"
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
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { DictAPI } from "dehaze-sdk-js";

const typeCode = ref("");
const dictName = ref("");
const list = ref<any[]>([]);
const showForm = ref(false);
const editId = ref(0);
const form = ref<any>({ name: "", value: "", sort: 0, status: 1 });
const saving = ref(false);

onLoad((options: any) => {
  typeCode.value = options?.typeCode || "";
  dictName.value = decodeURIComponent(options?.name || "");
  fetchList();
});

const fetchList = async () => {
  try {
    const res = await DictAPI.getDictPage({
      typeCode: typeCode.value,
      pageNum: 1,
      pageSize: 100,
    });
    list.value = res.list || [];
  } catch {}
};
const editItem = (item: any) => {
  if (item) {
    form.value = {
      name: item.name || "",
      value: item.value || "",
      sort: item.sort ?? 0,
      status: item.status ?? 1,
    };
    editId.value = item.id;
  } else {
    form.value = { name: "", value: "", sort: 0, status: 1 };
    editId.value = 0;
  }
  showForm.value = true;
};
const handleSave = async () => {
  saving.value = true;
  try {
    if (editId.value)
      await DictAPI.updateDict(editId.value, {
        ...form.value,
        id: editId.value,
        typeCode: typeCode.value,
      });
    else
      await DictAPI.addDict({ ...form.value, typeCode: typeCode.value });
    showForm.value = false;
    editId.value = 0;
    fetchList();
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};
const delItem = async (id: number) => {
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除该字典项吗？",
  });
  if (!res.confirm) return;
  try {
    await DictAPI.deleteDictByIds(String(id));
    fetchList();
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};
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
