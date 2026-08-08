<template>
  <view class="tree-node-wrapper">
    <!-- 当前节点 -->
    <view
      class="tree-node"
      :class="{ disabled: dataset.status === 0 }"
      :style="{ paddingLeft: (12 + depth * 20) + 'px' }"
      @click="handleClick"
    >
      <!-- 展开/收起按钮 -->
      <view
        v-if="hasChildren"
        class="expand-btn"
        :class="{ expanded: isExpanded }"
        @click.stop="handleToggleExpand"
      >
        <text v-if="isLoading" class="expand-loading">●</text>
        <text v-else>▶</text>
      </view>
      <view v-else class="expand-placeholder" />

      <!-- 信息区 -->
      <view class="node-info">
        <view class="node-header">
          <text class="node-name">{{ dataset.name }}</text>
          <u-tag :text="typeLabel" size="mini" />
          <u-tag
            v-if="dataset.status === 0"
            text="禁用"
            type="error"
            size="mini"
          />
        </view>
        <text class="node-desc">{{ dataset.description || "暂无描述" }}</text>
        <view class="node-stats">
          <text>图片: {{ fileCount }}</text>
          <text>{{ formattedDate }}</text>
        </view>
      </view>

      <!-- 操作按钮 -->
      <view class="node-actions">
        <u-button
          v-if="onAddChild"
          type="primary"
          size="mini"
          @click.stop="handleAddChild"
        >
          <text>+子</text>
        </u-button>
        <u-button
          v-if="onEdit"
          type="warning"
          size="mini"
          @click.stop="handleEdit"
        >
          <text>✎</text>
        </u-button>
        <u-button
          v-if="onDelete"
          type="error"
          size="mini"
          @click.stop="handleDelete"
        >
          <text>✕</text>
        </u-button>
      </view>
    </view>

    <!-- 递归渲染子节点 -->
    <template v-if="isExpanded && children && children.length > 0">
      <DatasetTreeNode
        v-for="child in children"
        :key="child.id"
        :dataset="child"
        :depth="depth + 1"
        :expanded-ids="expandedIds"
        :children-map="childrenMap"
        :children-loading="childrenLoading"
        @toggle-expand="(id: number) => $emit('toggleExpand', id)"
        @click="(ds: any) => $emit('click', ds)"
        @add-child="(ds: any) => $emit('addChild', ds)"
        @edit="(ds: any) => $emit('edit', ds)"
        @delete="(ds: any) => $emit('delete', ds)"
      />
    </template>

    <!-- 子节点加载中 -->
    <view
      v-if="isExpanded && isLoading && (!children || children.length === 0)"
      class="children-loading"
    >
      <text>加载中...</text>
    </view>

    <!-- 展开但无子节点 -->
    <view
      v-if="isExpanded && !isLoading && children && children.length === 0"
      class="empty-children"
    >
      <text>暂无子数据集</text>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { Dataset } from "dehaze-sdk-js";

const typeLabels: Record<string, string> = {
  training: "训练集",
  test: "测试集",
  user: "用户集",
  result: "结果集",
};

interface Props {
  dataset: Dataset;
  depth: number;
  expandedIds: number[];
  childrenMap: Record<number, Dataset[]>;
  childrenLoading: Record<number, boolean>;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: "toggleExpand", id: number): void;
  (e: "click", dataset: Dataset): void;
  (e: "addChild", dataset: Dataset): void;
  (e: "edit", dataset: Dataset): void;
  (e: "delete", dataset: Dataset): void;
}>();

const hasChildren = computed(() => props.dataset.hasChildren === true);
const isExpanded = computed(() => props.expandedIds.includes(props.dataset.id));
const isLoading = computed(() => !!props.childrenLoading[props.dataset.id]);
const children = computed(() => props.childrenMap[props.dataset.id]);

const typeLabel = computed(() => typeLabels[props.dataset.type] || props.dataset.type);
const fileCount = computed(
  () => props.dataset.statistics?.fileCount || props.dataset.total || 0
);

const formattedDate = computed(() => {
  const time = props.dataset.createTime;
  if (!time) return "-";
  const date = new Date(typeof time === "string" ? time : time);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return `${days}天前`;
  return date.toLocaleDateString("zh-CN");
});

const handleToggleExpand = () => emit("toggleExpand", props.dataset.id);
const handleClick = () => emit("click", props.dataset);
const handleAddChild = () => emit("addChild", props.dataset);
const handleEdit = () => emit("edit", props.dataset);
const handleDelete = () => emit("delete", props.dataset);
</script>

<script lang="ts">
export default {
  name: "DatasetTreeNode",
};
</script>

<style lang="scss" scoped>
.tree-node-wrapper {
  width: 100%;
}

.tree-node {
  display: flex;
  align-items: center;
  padding: 16rpx 20rpx;
  background: #ffffff;
  border-bottom: 2rpx solid #f3f4f6;
  transition: background 0.2s;

  &:active {
    background: #f9fafb;
  }

  &.disabled {
    opacity: 0.6;
  }
}

.expand-btn {
  width: 40rpx;
  height: 40rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20rpx;
  color: #9ca3af;
  flex-shrink: 0;

  &.expanded {
    transform: rotate(90deg);
  }
}

.expand-placeholder {
  width: 40rpx;
  flex-shrink: 0;
}

.expand-loading {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.node-info {
  flex: 1;
  min-width: 0;
  margin-right: 16rpx;
}

.node-header {
  display: flex;
  align-items: center;
  gap: 12rpx;
  margin-bottom: 8rpx;
}

.node-name {
  font-size: 28rpx;
  font-weight: 600;
  color: #1f2937;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.node-desc {
  display: block;
  font-size: 24rpx;
  color: #6b7280;
  margin-bottom: 8rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.node-stats {
  display: flex;
  gap: 24rpx;
  font-size: 22rpx;
  color: #9ca3af;
}

.node-actions {
  display: flex;
  gap: 8rpx;
  flex-shrink: 0;
}

.children-loading,
.empty-children {
  padding: 16rpx 32rpx;
  font-size: 24rpx;
  color: #9ca3af;
  text-align: center;
}
</style>
