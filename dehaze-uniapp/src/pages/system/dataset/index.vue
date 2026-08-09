<template>
  <PageLayout level="L2" title="数据集管理">
    <view class="page-body">
      <!-- 搜索栏 -->
      <view class="search-bar">
        <input
          class="search-input"
          v-model="keyword"
          placeholder="搜索数据集名称"
          confirm-type="search"
          @confirm="handleSearch"
        />
      </view>

      <!-- 操作栏 -->
      <view class="action-bar">
        <button class="btn btn-primary btn-sm" @click="handleAddRoot">
          新增数据集
        </button>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <view class="loading-spinner" />
        <text class="loading-text">加载中...</text>
      </view>

      <!-- 数据集树形列表 -->
      <view v-else-if="list.length > 0" class="dataset-tree">
        <view v-for="item in list" :key="item.id">
          <DatasetTreeNode
            :dataset="item"
            :depth="0"
            :expanded-ids="expandedIds"
            :children-map="childrenMap"
            :children-loading="childrenLoading"
            @toggle-expand="toggleExpand"
            @click="handleItemClick"
            @add-child="handleAddChild"
            @edit="handleEdit"
            @delete="handleDelete"
          />
        </view>
      </view>

      <!-- 空状态 -->
      <view v-else class="empty-tip">暂无数据集</view>

      <!-- 加载更多 -->
      <view
        v-if="hasMore && list.length > 0"
        class="load-more"
        @click="loadMore"
      >
        <text>加载更多</text>
      </view>
    </view>

    <!-- 新增/编辑弹窗 -->
    <Popup :show="showForm" mode="center" round @close="closeForm">
      <view class="popup-content">
        <view class="popup-title">{{
          form.id ? "编辑数据集" : "新增数据集"
        }}</view>
        <view class="form-row">
          <text class="form-label">上级数据集</text>
          <input
            class="form-input"
            :value="parentLabel"
            disabled
            placeholder="根数据集"
            @click="showParentSelect = true"
          />
        </view>
        <view class="form-row">
          <text class="form-label">名称 *</text>
          <input
            class="form-input"
            v-model="form.name"
            placeholder="数据集名称"
          />
        </view>
        <view class="form-row form-row-block">
          <text class="form-label">类型</text>
          <view class="type-selector">
            <view
              v-for="opt in typeOptions"
              :key="opt.value"
              class="tag"
              :class="form.type === opt.value ? 'tag-primary' : 'tag-info'"
              @click="form.type = opt.value"
            >
              {{ opt.label }}
            </view>
          </view>
        </view>
        <view class="form-row form-row-block">
          <text class="form-label">描述</text>
          <textarea
            class="form-textarea"
            v-model="form.description"
            placeholder="描述（选填）"
            maxlength="200"
          />
        </view>
        <view class="form-row">
          <text class="form-label">状态</text>
          <switch
            :checked="form.status === 1"
            @change="(e: any) => (form.status = e.detail.value ? 1 : 0)"
          />
        </view>
        <view v-if="formError" class="form-error">{{ formError }}</view>
        <button class="btn btn-primary" :disabled="saving" @click="handleSave">
          保存
        </button>
      </view>
    </Popup>

    <!-- 父级选择弹窗 -->
    <Popup
      :show="showParentSelect"
      mode="center"
      round
      @close="showParentSelect = false"
    >
      <view class="popup-content">
        <view class="popup-title">选择上级数据集</view>
        <view
          class="select-item"
          :class="{ active: form.parentId === 0 }"
          @click="selectParent(0, '根数据集')"
        >
          <text>根数据集</text>
        </view>
        <view
          v-for="opt in datasetOptions"
          :key="opt.value"
          class="select-item"
          :class="{ active: form.parentId === opt.value }"
          @click="selectParent(Number(opt.value), opt.label)"
        >
          <text>{{ opt.label }}</text>
        </view>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import DatasetTreeNode from "./components/DatasetTreeNode.vue";
import { usePagedList } from "@/composables/usePagedList";
import { DatasetAPI } from "dehaze-sdk-js";
import type { Dataset, OptionType } from "dehaze-sdk-js";

// ==================== 类型选项 ====================
const typeOptions = [
  { value: "training", label: "训练集" },
  { value: "test", label: "测试集" },
  { value: "user", label: "用户集" },
  { value: "result", label: "结果集" },
];

// ==================== 列表状态 ====================
const { list, keyword, hasMore, loading, fetchList, handleSearch, loadMore } =
  usePagedList<Dataset>({
    fetcher: (p) =>
      DatasetAPI.getList({
        pageNum: p.pageNum,
        pageSize: 10,
        keyword: p.keyword,
      }).then((r) => (r.list as Dataset[]) || []),
  });

const expandedIds = ref<number[]>([]);
const childrenMap = ref<Record<number, Dataset[]>>({});
const childrenLoading = ref<Record<number, boolean>>({});

// ==================== 表单状态 ====================
const showForm = ref(false);
const form = ref({
  id: 0,
  parentId: 0,
  name: "",
  type: "user",
  description: "",
  status: 1,
});
const formError = ref("");
const saving = ref(false);
const parentLabel = ref("根数据集");
const showParentSelect = ref(false);
const datasetOptions = ref<OptionType[]>([]);

// ==================== 数据加载 ====================
const loadOptions = async () => {
  try {
    const options = await DatasetAPI.getOptions();
    datasetOptions.value = options || [];
  } catch {
    // 忽略
  }
};

// ==================== 树形展开 ====================
const toggleExpand = async (id: number) => {
  const idx = expandedIds.value.indexOf(id);
  if (idx >= 0) {
    expandedIds.value.splice(idx, 1);
    return;
  }
  expandedIds.value.push(id);
  if (!childrenMap.value[id]) {
    childrenLoading.value[id] = true;
    try {
      const children = await DatasetAPI.getChildren(id);
      childrenMap.value[id] = children || [];
    } finally {
      childrenLoading.value[id] = false;
    }
  }
};

// ==================== CRUD 操作 ====================
const handleAddRoot = () => {
  form.value = {
    id: 0,
    parentId: 0,
    name: "",
    type: "user",
    description: "",
    status: 1,
  };
  parentLabel.value = "根数据集";
  formError.value = "";
  showForm.value = true;
};

const handleAddChild = (parent: Dataset) => {
  form.value = {
    id: 0,
    parentId: parent.id,
    name: "",
    type: "user",
    description: "",
    status: 1,
  };
  parentLabel.value = parent.name;
  formError.value = "";
  showForm.value = true;
};

const handleEdit = (dataset: Dataset) => {
  form.value = {
    id: dataset.id,
    parentId: dataset.parentId ?? 0,
    name: dataset.name,
    type: dataset.type,
    description: dataset.description || "",
    status: dataset.status ?? 1,
  };
  parentLabel.value =
    datasetOptions.value.find((o) => o.value === dataset.parentId)?.label ||
    "根数据集";
  formError.value = "";
  showForm.value = true;
};

const handleDelete = async (dataset: Dataset) => {
  const res = await uni.showModal({
    title: "确认删除",
    content: `确定要删除数据集「${dataset.name}」吗？此操作不可恢复，子数据集和图片也将被删除。`,
    confirmText: "删除",
    confirmColor: "#ef4444",
  });
  if (!res.confirm) return;
  try {
    await DatasetAPI.deleteById(dataset.id);
    uni.showToast({ title: "删除成功", icon: "success" });
    fetchList(true);
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};

const handleItemClick = (dataset: Dataset) => {
  uni.navigateTo({ url: `/pages/dataset/index?datasetId=${dataset.id}` });
};

// ==================== 表单操作 ====================
const selectParent = (parentId: number, label: string) => {
  form.value.parentId = parentId;
  parentLabel.value = label;
  showParentSelect.value = false;
};

const closeForm = () => {
  showForm.value = false;
  formError.value = "";
};

const handleSave = async () => {
  if (!form.value.name.trim()) {
    formError.value = "请输入数据集名称";
    return;
  }
  saving.value = true;
  try {
    if (form.value.id) {
      await DatasetAPI.update(form.value.id, {
        type: form.value.type,
        name: form.value.name,
        description: form.value.description,
        status: String(form.value.status),
      });
      uni.showToast({ title: "更新成功", icon: "success" });
    } else {
      await DatasetAPI.add({
        parentId: form.value.parentId,
        type: form.value.type,
        name: form.value.name,
        description: form.value.description,
        status: String(form.value.status),
      });
      uni.showToast({ title: "创建成功", icon: "success" });
    }
    showForm.value = false;
    loadOptions();
    fetchList(true);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};

// ==================== 初始化 ====================
onMounted(() => {
  fetchList(true);
  loadOptions();
});
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}

.search-bar {
  margin-bottom: 20rpx;

  .search-input {
    width: 100%;
    box-sizing: border-box;
    padding: 14rpx 20rpx;
    font-size: 28rpx;
    background: $color-bg-secondary;
    border-radius: $radius-md;
  }
}

.action-bar {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 20rpx;
}

.dataset-tree {
  min-height: 400rpx;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 160rpx 0;
}
.loading-text {
  margin-top: 16rpx;
  font-size: 28rpx;
  color: $color-text-placeholder;
}

.empty-tip {
  text-align: center;
  padding: 160rpx 0;
  color: $color-text-secondary;
}

.load-more {
  text-align: center;
  padding: 32rpx;
  color: $color-text-placeholder;
  font-size: 28rpx;
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

.form-error {
  padding: 16rpx;
  margin-bottom: 20rpx;
  color: $color-danger;
  font-size: 26rpx;
  background: $color-danger-bg;
  border-radius: $radius-sm;
}

.tag {
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-primary {
  color: $color-primary;
  background: $color-primary-bg;
}
.tag-info {
  color: $color-text-secondary;
  background: $color-bg-secondary;
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

.type-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 16rpx;
}

.form-row {
  display: flex;
  align-items: center;
  padding: 20rpx 0;
  border-bottom: 1rpx solid $color-border;

  .form-label {
    width: 180rpx;
    flex-shrink: 0;
    font-size: 28rpx;
    color: $color-text-secondary;
  }

  .form-input {
    flex: 1;
    font-size: 28rpx;
  }
}
.form-row-block {
  flex-direction: column;
  align-items: stretch;

  .form-label {
    width: auto;
    margin-bottom: 12rpx;
  }
}
.form-textarea {
  width: 100%;
  box-sizing: border-box;
  min-height: 120rpx;
  padding: 12rpx;
  font-size: 28rpx;
  border: 1rpx solid $color-border;
  border-radius: $radius-sm;
}

.select-item {
  padding: 20rpx 32rpx;
  border-bottom: 2rpx solid $color-border-light;
  font-size: 28rpx;
  color: $color-text-primary;

  &.active {
    color: $color-primary;
    background: $color-primary-bg;
  }
}
</style>
