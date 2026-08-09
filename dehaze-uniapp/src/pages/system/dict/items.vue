<template>
  <PageLayout level="L2" :title="dictName + ' - 字典项'">
    <view class="page-body">
      <view class="list-row" v-for="item in list" :key="item.id">
        <text class="cell">{{ item.name }}</text>
        <text class="cell">{{ item.value }}</text>
        <view class="cell">
          <view
            class="tag"
            :class="item.status === 1 ? 'tag-success' : 'tag-danger'"
          >
            {{ item.status === 1 ? "启用" : "禁用" }}
          </view>
        </view>
        <view class="cell row-actions">
          <SvgIcon v-if="canEdit" name="edit-pen" @click="editItem(item)" />
          <SvgIcon
            v-if="canDelete"
            name="trash"
            color="#ef4444"
            @click="delItem(item.id)"
          />
        </view>
      </view>
      <view v-if="list.length === 0" class="empty-tip">暂无字典项</view>
    </view>
    <FabButton v-if="canAdd" @click="editItem(null)">
      <SvgIcon name="plus" color="#fff" size="24" />
    </FabButton>
    <Popup :show="showForm" mode="center" round @close="showForm = false">
      <view class="popup-body">
        <view class="popup-title">{{
          editId ? "编辑字典项" : "新增字典项"
        }}</view>
        <view class="form-row">
          <text class="form-label">标签</text>
          <input
            class="form-input"
            v-model="form.name"
            placeholder="显示标签"
          />
        </view>
        <view class="form-row">
          <text class="form-label">值</text>
          <input class="form-input" v-model="form.value" placeholder="字典值" />
        </view>
        <view class="form-row">
          <text class="form-label">排序</text>
          <input
            class="form-input"
            type="number"
            v-model.number="form.sort"
            placeholder="排序"
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
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import FabButton from "@/components/common/FabButton.vue";
import Popup from "@/components/common/Popup.vue";
import { DictAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:dict:data:add"));
const canEdit = computed(() => authStore.hasPerm("sys:dict:data:edit"));
const canDelete = computed(() => authStore.hasPerm("sys:dict:data:delete"));

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
    else await DictAPI.addDict({ ...form.value, typeCode: typeCode.value });
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
