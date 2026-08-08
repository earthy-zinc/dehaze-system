<template>
  <PageLayout level="L2" title="推荐管理">
    <view class="page-body">
      <view class="stats-bar">
        <view class="stat-item">
          <text class="stat-value">{{ rules.length }}</text>
          <text class="stat-label">总规则数</text>
        </view>
        <view class="stat-item">
          <text class="stat-value">{{ enabledCount }}</text>
          <text class="stat-label">生效中</text>
        </view>
      </view>

      <view v-if="loading" class="loading-container">
        <up-loading-icon mode="circle" size="40" />
        <text class="loading-text">加载中...</text>
      </view>

      <view v-else-if="rules.length > 0" class="list">
        <view v-for="rule in rules" :key="rule.id" class="rule-card">
          <view class="rule-top">
            <text class="rule-name">{{ rule.ruleName }}</text>
            <text :class="['status-tag', rule.enabled ? 's-enabled' : 's-disabled']">
              {{ rule.enabled ? '生效中' : '已禁用' }}
            </text>
          </view>
          <view class="rule-meta">
            <text v-if="rule.sceneType" class="meta-tag">场景: {{ rule.sceneType }}</text>
            <text class="meta-tag">权重: {{ rule.weight }}</text>
          </view>
          <view v-if="rule.algorithmIds && rule.algorithmIds.length > 0" class="rule-algos">
            <text class="algo-label">关联算法: </text>
            <text>{{ rule.algorithmIds.join(', ') }}</text>
          </view>
          <view class="rule-actions">
            <text class="action-btn" @click="editRule(rule)">编辑</text>
            <text class="action-btn danger" @click="handleDelete(rule)">删除</text>
          </view>
        </view>
      </view>

      <view v-else class="empty-state">
        <up-empty mode="list" text="暂无推荐规则" />
      </view>
    </view>

    <!-- 编辑弹窗 -->
    <u-popup :show="showForm" mode="bottom" round="24" @close="showForm = false">
      <view class="popup-content">
        <text class="popup-title">编辑推荐规则</text>
        <u-form :model="form">
          <u-form-item label="规则名称">
            <u-input v-model="form.ruleName" placeholder="请输入规则名称" />
          </u-form-item>
          <u-form-item label="场景类型">
            <u-input v-model="form.sceneType" placeholder="如: urban, landscape" />
          </u-form-item>
          <u-form-item label="关联算法ID">
            <u-input v-model="form.algorithmIds" placeholder="逗号分隔, 如: 1,2,3" />
          </u-form-item>
          <u-form-item label="匹配权重">
            <u-input v-model="form.weight" type="number" placeholder="0-100" />
          </u-form-item>
          <u-form-item label="启用规则">
            <u-switch :checked="form.enabled" @change="(val: boolean) => (form.enabled = val)" />
          </u-form-item>
        </u-form>
        <view class="popup-footer">
          <u-button text="取消" @click="showForm = false" />
          <u-button text="保存" type="primary" @click="handleSave" :loading="saving" />
        </view>
      </view>
    </u-popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import { RecommendationAPI } from "dehaze-sdk-js";
import type { RecommendationRule } from "dehaze-sdk-js";

const rules = ref<RecommendationRule[]>([]);
const loading = ref(false);
const showForm = ref(false);
const editingRule = ref<RecommendationRule | null>(null);
const saving = ref(false);

const form = ref({
  ruleName: "",
  sceneType: "",
  algorithmIds: "",
  weight: "1",
  enabled: true,
});

const enabledCount = computed(() => rules.value.filter((r) => r.enabled).length);

async function fetchRules() {
  loading.value = true;
  try {
    rules.value = await RecommendationAPI.getRules();
  } catch {
    uni.showToast({ title: "加载失败", icon: "error" });
  } finally {
    loading.value = false;
  }
}

function editRule(rule: RecommendationRule) {
  editingRule.value = rule;
  form.value = {
    ruleName: rule.ruleName,
    sceneType: rule.sceneType || "",
    algorithmIds: rule.algorithmIds?.join(",") || "",
    weight: String(rule.weight ?? 1),
    enabled: rule.enabled ?? true,
  };
  showForm.value = true;
}

async function handleSave() {
  if (!form.value.ruleName || !editingRule.value?.id) {
    uni.showToast({ title: "请填写规则名称", icon: "none" });
    return;
  }
  saving.value = true;
  try {
    const algoIds = form.value.algorithmIds
      ? form.value.algorithmIds
          .split(",")
          .map((s: string) => Number(s.trim()))
          .filter((n: number) => !isNaN(n))
      : [];
    await RecommendationAPI.updateRule(editingRule.value.id, {
      id: editingRule.value.id,
      ruleName: form.value.ruleName,
      sceneType: form.value.sceneType,
      algorithmIds: algoIds,
      weight: Number(form.value.weight) || 1,
      enabled: form.value.enabled,
    });
    uni.showToast({ title: "保存成功", icon: "success" });
    showForm.value = false;
    fetchRules();
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
}

async function handleDelete(rule: RecommendationRule) {
  const res = await uni.showModal({
    title: "确认删除",
    content: `确定要删除规则「${rule.ruleName}」吗？`,
    confirmColor: "#ef4444",
  });
  if (!res.confirm) return;
  try {
    await RecommendationAPI.updateRule(rule.id!, {
      ...rule,
      enabled: false,
    });
    uni.showToast({ title: "已删除", icon: "success" });
    fetchRules();
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
}

fetchRules();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
  padding-bottom: 60rpx;
}

.stats-bar {
  display: flex;
  gap: 24rpx;
  margin-bottom: $spacing-md;
}
.stat-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20rpx;
  background: #fff;
  border-radius: $radius-lg;
}
.stat-value {
  font-size: 40rpx;
  font-weight: 700;
  color: $color-primary;
}
.stat-label {
  font-size: $font-xs;
  color: $color-text-secondary;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.rule-card {
  background: #fff;
  border-radius: $radius-lg;
  padding: 24rpx;
  border-left: 6rpx solid $color-primary;
  box-shadow: $shadow-sm;
}
.rule-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12rpx;
}
.rule-name {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.status-tag {
  font-size: $font-xs;
  font-weight: 500;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
  flex-shrink: 0;
}
.s-enabled {
  color: #10b981;
  background: #ecfdf5;
}
.s-disabled {
  color: #9ca3af;
  background: #f3f4f6;
}

.rule-meta {
  display: flex;
  gap: 12rpx;
  margin-bottom: 8rpx;
}
.meta-tag {
  font-size: $font-xs;
  color: $color-primary;
  background: $color-primary-bg;
  padding: 4rpx 12rpx;
  border-radius: 6rpx;
}
.rule-algos {
  margin-bottom: 12rpx;
  font-size: $font-xs;
  color: $color-text-secondary;
}
.algo-label {
  font-weight: 500;
}

.rule-actions {
  display: flex;
  gap: 16rpx;
  justify-content: flex-end;
}
.action-btn {
  font-size: $font-sm;
  color: $color-primary;
  &.danger {
    color: #ef4444;
  }
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.loading-text {
  margin-top: 24rpx;
  font-size: $font-md;
  color: $color-text-placeholder;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.popup-content {
  padding: 32rpx;
}
.popup-title {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
  display: block;
  margin-bottom: 24rpx;
}
.popup-footer {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
}
</style>
