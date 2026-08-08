<template>
  <PageLayout level="L2" title="数据集管理">
    <view class="page-body">
      <!-- 搜索栏 -->
      <view class="search-bar">
        <u-search
          v-model="keyword"
          placeholder="搜索数据集名称"
          @search="handleSearch"
          @clear="handleSearch"
        />
      </view>

      <!-- 操作栏 -->
      <view class="action-bar">
        <u-button type="primary" size="small" @click="handleAddRoot">
          <text>新增数据集</text>
        </u-button>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <u-loading-icon mode="circle" size="40" color="#14b8a6" />
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
      <view v-else class="empty-container">
        <u-empty text="暂无数据集" />
      </view>

      <!-- 加载更多 -->
      <view v-if="hasMore && list.length > 0" class="load-more" @click="loadMore">
        <text>加载更多</text>
      </view>
    </view>

    <!-- 新增/编辑弹窗 -->
    <u-popup :show="showForm" @close="closeForm" round>
      <view class="popup-content">
        <view class="popup-title">{{ form.id ? "编辑数据集" : "新增数据集" }}</view>
        <u-form :model="form" label-width="120rpx">
          <u-form-item label="上级数据集">
            <u-input
              :value="parentLabel"
              disabled
              placeholder="根数据集"
              @click="showParentSelect = true"
            />
          </u-form-item>
          <u-form-item label="名称" required>
            <u-input v-model="form.name" placeholder="数据集名称" />
          </u-form-item>
          <u-form-item label="类型">
            <view class="type-selector">
              <u-tag
                v-for="opt in typeOptions"
                :key="opt.value"
                :text="opt.label"
                :type="form.type === opt.value ? 'primary' : 'info'"
                @click="form.type = opt.value"
              />
            </view>
          </u-form-item>
          <u-form-item label="描述">
            <u-input
              v-model="form.description"
              type="textarea"
              placeholder="描述（选填）"
              :maxlength="200"
            />
          </u-form-item>
          <u-form-item label="状态">
            <u-switch
              :checked="form.status === 1"
              @change="(val: boolean) => (form.status = val ? 1 : 0)"
            />
          </u-form-item>
        </u-form>
        <view v-if="formError" class="form-error">{{ formError }}</view>
        <u-button type="primary" @click="handleSave" :loading="saving">保存</u-button>
      </view>
    </u-popup>

    <!-- 父级选择弹窗 -->
    <u-popup :show="showParentSelect" @close="showParentSelect = false" round>
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
    </u-popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import DatasetTreeNode from "./components/DatasetTreeNode.vue";
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
const list = ref<Dataset[]>([]);
const keyword = ref("");
const pageNum = ref(1);
const hasMore = ref(false);
const loading = ref(false);
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
const fetchList = async (reset = false) => {
  if (reset) {
    pageNum.value = 1;
    list.value = [];
  }
  loading.value = true;
  try {
    const res = await DatasetAPI.getList({
      pageNum: pageNum.value,
      pageSize: 10,
      keyword: keyword.value || undefined,
    });
    const records = (res.list as Dataset[]) || [];
    if (reset) list.value = records;
    else list.value.push(...records);
    hasMore.value = records.length === 10;
    pageNum.value++;
  } finally {
    loading.value = false;
  }
};

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

// ==================== 搜索/翻页 ====================
const handleSearch = () => fetchList(true);
const loadMore = () => fetchList();

// ==================== CRUD 操作 ====================
const handleAddRoot = () => {
  form.value = { id: 0, parentId: 0, name: "", type: "user", description: "", status: 1 };
  parentLabel.value = "根数据集";
  formError.value = "";
  showForm.value = true;
};

const handleAddChild = (parent: Dataset) => {
  form.value = { id: 0, parentId: parent.id, name: "", type: "user", description: "", status: 1 };
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
  parentLabel.value = datasetOptions.value.find((o) => o.value === dataset.parentId)?.label || "根数据集";
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
  color: #9ca3af;
}

.empty-container {
  padding: 160rpx 0;
}

.load-more {
  text-align: center;
  padding: 32rpx;
  color: #9ca3af;
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
  color: #ef4444;
  font-size: 26rpx;
  background: #fef2f2;
  border-radius: 8rpx;
}

.type-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 16rpx;
}

.select-item {
  padding: 20rpx 32rpx;
  border-bottom: 2rpx solid #f3f4f6;
  font-size: 28rpx;
  color: #374151;

  &.active {
    color: #14b8a6;
    background: #f0fdfa;
  }
}
</style>
