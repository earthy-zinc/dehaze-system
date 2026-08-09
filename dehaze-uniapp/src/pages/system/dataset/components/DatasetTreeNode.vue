<template>
  <view class="tree-node-wrapper">
    <!-- 当前节点 -->
    <view
      class="tree-node"
      :class="{ disabled: dataset.status === 0 }"
      :style="{ paddingLeft: 12 + depth * 20 + 'px' }"
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
          <view class="tag tag-info tag-sm">{{ typeLabel }}</view>
          <view v-if="dataset.status === 0" class="tag tag-danger tag-sm"
            >禁用</view
          >
        </view>
        <text class="node-desc">{{ dataset.description || "暂无描述" }}</text>
        <view class="node-stats">
          <text>图片: {{ fileCount }}</text>
          <text>{{ formattedDate }}</text>
        </view>
      </view>

      <!-- 操作按钮 -->
      <view class="node-actions">
        <button class="btn btn-primary btn-sm" @click.stop="handleAddChild">
          +子
        </button>
        <button class="btn btn-warning btn-sm" @click.stop="handleEdit">
          ✎
        </button>
        <button class="btn btn-danger btn-sm" @click.stop="handleDelete">
          ✕
        </button>
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

const typeLabel = computed(
  () => typeLabels[props.dataset.type] || props.dataset.type
);
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
  background: $color-white;
  border-bottom: 2rpx solid $color-border-light;
  transition: background 0.2s;

  &:active {
    background: $color-bg-primary;
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
  color: $color-text-placeholder;
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
  color: $color-text-primary;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.node-desc {
  display: block;
  font-size: 24rpx;
  color: $color-text-secondary;
  margin-bottom: 8rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.node-stats {
  display: flex;
  gap: 24rpx;
  font-size: 22rpx;
  color: $color-text-placeholder;
}

.node-actions {
  display: flex;
  gap: 8rpx;
  flex-shrink: 0;
}

.tag {
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-sm {
  padding: 2rpx 10rpx;
}
.tag-primary {
  color: $color-primary;
  background: $color-primary-bg;
}
.tag-danger {
  color: $color-danger;
  background: $color-danger-bg;
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
.btn-warning {
  color: $color-white;
  background: $color-warning;
}
.btn-danger {
  color: $color-white;
  background: $color-danger;
}

.children-loading,
.empty-children {
  padding: 16rpx 32rpx;
  font-size: 24rpx;
  color: $color-text-placeholder;
  text-align: center;
}
</style>
