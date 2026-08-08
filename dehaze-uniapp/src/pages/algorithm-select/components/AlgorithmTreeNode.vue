<template>
  <view>
    <view
      class="tree-node"
      :class="{
        'tree-node--leaf': isLeaf,
        'tree-node--selectable': true,
      }"
      :style="{ paddingLeft: 32 + level * 32 + 'rpx' }"
      @click="handleClick"
    >
      <!-- 展开/收起图标 -->
      <view v-if="hasChildren" class="tree-node__expand" @click.stop="emit('toggleExpand', node.id)">
        <text :class="{ expanded: isExpanded }">{{ isExpanded ? '▼' : '▶' }}</text>
      </view>
      <view v-else class="tree-node__icon">
        <text>⚡</text>
      </view>

      <!-- 节点内容 -->
      <view class="tree-node__content">
        <view class="tree-node__header">
          <text class="tree-node__name">{{ node.name }}</text>
          <text v-if="node.type" class="tree-node__type">{{ node.type }}</text>
        </view>
      </view>

      <!-- 叶子节点操作按钮 -->
      <view v-if="isLeaf" class="tree-node__actions">
        <view class="tree-node__action" @click.stop="emit('toggleFavorite', node.id)">
          <SvgIcon
            :name="favorite ? 'star-fill' : 'star'"
            size="16"
            :color="favorite ? '#f59e0b' : '#9ca3af'"
          />
        </view>
        <view class="tree-node__action" @click.stop="emit('showDetail', node)">
          <text class="tree-node__detail-btn">详情</text>
        </view>
        <view
          class="tree-node__action"
          :class="{ 'tree-node__action--compare': inCompare }"
          @click.stop="emit('toggleCompare', node)"
        >
          <text class="tree-node__compare-btn" :class="{ active: inCompare }">对比</text>
        </view>
      </view>
    </view>

    <!-- 子节点 -->
    <view v-if="hasChildren && isExpanded" class="tree-node__children">
      <AlgorithmTreeNode
        v-for="child in node.children"
        :key="child.id"
        :node="child"
        :level="level + 1"
        :expanded-keys="expandedKeys"
        :favorite-ids="favoriteIds"
        :compare-list="compareList"
        @toggle-expand="(id: number) => emit('toggleExpand', id)"
        @select="(n: AlgorithmSelectNodeVO) => emit('select', n)"
        @toggle-favorite="(id: number) => emit('toggleFavorite', id)"
        @show-detail="(n: AlgorithmSelectNodeVO) => emit('showDetail', n)"
        @toggle-compare="(n: AlgorithmSelectNodeVO) => emit('toggleCompare', n)"
      />
    </view>
  </view>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import type { AlgorithmSelectNodeVO } from "dehaze-sdk-js";
import SvgIcon from "@/components/SvgIcon/index.vue";

const props = defineProps<{
  node: AlgorithmSelectNodeVO;
  level: number;
  expandedKeys: Set<number>;
  favoriteIds: Set<number>;
  compareList: AlgorithmSelectNodeVO[];
}>();

const emit = defineEmits<{
  toggleExpand: [id: number];
  select: [node: AlgorithmSelectNodeVO];
  toggleFavorite: [id: number];
  showDetail: [node: AlgorithmSelectNodeVO];
  toggleCompare: [node: AlgorithmSelectNodeVO];
}>();

const hasChildren = computed(() => !!(props.node.children && props.node.children.length > 0));
const isExpanded = computed(() => props.expandedKeys.has(props.node.id));
const isLeaf = computed(() => !hasChildren.value && props.node.leaf);
const favorite = computed(() => props.favoriteIds.has(props.node.id));
const inCompare = computed(() => props.compareList.some((c) => c.id === props.node.id));

function handleClick() {
  if (hasChildren.value) {
    emit("toggleExpand", props.node.id);
  } else {
    emit("select", props.node);
  }
}
</script>

<style lang="scss" scoped>
.tree-node {
  display: flex;
  align-items: center;
  padding: 24rpx 32rpx;
  background: #fff;
  border-bottom: 2rpx solid #f5f5f5;
}

.tree-node--selectable:active {
  background: #f9fafb;
}

.tree-node__expand {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48rpx;
  height: 48rpx;
  margin-right: 16rpx;
  font-size: 24rpx;
  color: #6b7280;
  flex-shrink: 0;

  .expanded {
    color: #3b82f6;
  }
}

.tree-node__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48rpx;
  height: 48rpx;
  margin-right: 16rpx;
  font-size: 24rpx;
  color: #f59e0b;
  flex-shrink: 0;
}

.tree-node__content {
  flex: 1;
  min-width: 0;
}

.tree-node__header {
  display: flex;
  gap: 12rpx;
  align-items: center;
}

.tree-node__name {
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 30rpx;
  font-weight: 500;
  color: #1f2937;
  white-space: nowrap;
}

.tree-node__type {
  flex-shrink: 0;
  padding: 4rpx 12rpx;
  font-size: 20rpx;
  color: #6366f1;
  background: #eef2ff;
  border-radius: 8rpx;
}

.tree-node__actions {
  display: flex;
  flex-shrink: 0;
  gap: 4rpx;
  align-items: center;
  margin-left: 12rpx;
}

.tree-node__action {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 56rpx;
  height: 56rpx;
  border-radius: 50%;

  &:active {
    background: #f3f4f6;
  }
}

.tree-node__action--compare {
  background: #ede9fe;
}

.tree-node__detail-btn {
  font-size: 22rpx;
  color: #6b7280;
}

.tree-node__compare-btn {
  font-size: 20rpx;
  color: #9ca3af;

  &.active {
    color: #8b5cf6;
  }
}

.tree-node__children {
  background: #fafbfc;
}
</style>
