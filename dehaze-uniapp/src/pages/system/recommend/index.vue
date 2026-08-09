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
        <view class="loading-spinner" />
        <text class="loading-text">加载中...</text>
      </view>

      <view v-else-if="rules.length > 0" class="list">
        <view v-for="rule in rules" :key="rule.id" class="rule-card">
          <view class="rule-top">
            <text class="rule-name">{{ rule.ruleName }}</text>
            <text
              :class="['status-tag', rule.enabled ? 's-enabled' : 's-disabled']"
            >
              {{ rule.enabled ? "生效中" : "已禁用" }}
            </text>
          </view>
          <view class="rule-meta">
            <text v-if="rule.sceneType" class="meta-tag"
              >场景: {{ rule.sceneType }}</text
            >
            <text class="meta-tag">权重: {{ rule.weight }}</text>
          </view>
          <view
            v-if="rule.algorithmIds && rule.algorithmIds.length > 0"
            class="rule-algos"
          >
            <text class="algo-label">关联算法: </text>
            <text>{{ rule.algorithmIds.join(", ") }}</text>
          </view>
          <view v-if="canEdit" class="rule-actions">
            <text class="action-btn" @click="editRule(rule)">编辑</text>
            <text class="action-btn danger" @click="handleDelete(rule)"
              >删除</text
            >
          </view>
        </view>
      </view>

      <view v-else class="empty-tip">暂无推荐规则</view>
    </view>

    <!-- 编辑弹窗 -->
    <Popup :show="showForm" mode="bottom" round @close="showForm = false">
      <view class="popup-content">
        <text class="popup-title">编辑推荐规则</text>
        <view class="form-item">
          <text class="form-label">规则名称</text>
          <input
            class="form-input"
            v-model="form.ruleName"
            placeholder="请输入规则名称"
          />
        </view>
        <view class="form-item">
          <text class="form-label">场景类型</text>
          <input
            class="form-input"
            v-model="form.sceneType"
            placeholder="如: urban, landscape"
          />
        </view>
        <view class="form-item">
          <text class="form-label">关联算法ID</text>
          <input
            class="form-input"
            v-model="form.algorithmIds"
            placeholder="逗号分隔, 如: 1,2,3"
          />
        </view>
        <view class="form-item">
          <text class="form-label">匹配权重</text>
          <input
            class="form-input"
            v-model="form.weight"
            type="number"
            placeholder="0-100"
          />
        </view>
        <view class="form-item switch-row">
          <text class="form-label">启用规则</text>
          <switch
            :checked="form.enabled"
            @change="(e: any) => (form.enabled = e.detail.value)"
          />
        </view>
        <view class="popup-footer">
          <button class="btn btn-default" @click="showForm = false">
            取消
          </button>
          <button
            class="btn btn-primary"
            :disabled="saving"
            @click="handleSave"
          >
            保存
          </button>
        </view>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import { RecommendationAPI } from "dehaze-sdk-js";
import type { RecommendationRule } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const auth = useAuthStore();
const canEdit = computed(() => auth.hasPerm("sys:recommendation:rule:edit"));

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

const enabledCount = computed(
  () => rules.value.filter((r) => r.enabled).length
);

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
  background: $color-white;
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
  background: $color-white;
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
  color: $color-success;
  background: $color-success-bg;
}
.s-disabled {
  color: $color-text-placeholder;
  background: $color-bg-secondary;
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
    color: $color-danger;
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
.empty-tip {
  text-align: center;
  padding: 80rpx 0;
  color: $color-text-secondary;
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
.form-item {
  padding: 16rpx 0;
  border-bottom: 1rpx solid $color-border;
  &:last-child {
    border-bottom: none;
  }
}
.form-label {
  display: block;
  margin-bottom: 12rpx;
  font-size: $font-sm;
  color: $color-text-secondary;
}
.form-input {
  width: 100%;
  box-sizing: border-box;
  padding: 14rpx 20rpx;
  font-size: $font-md;
  background: $color-bg-primary;
  border-radius: $radius-md;
}
.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  .form-label {
    margin-bottom: 0;
  }
}
.popup-footer {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
}
.btn {
  flex: 1;
  padding: 12rpx 20rpx;
  border-radius: $radius-sm;
  font-size: $font-sm;
  line-height: 1.6;
  &::after {
    border: none;
  }
}
.btn-primary {
  color: $color-white;
  background: $color-primary;
}
.btn-default {
  color: $color-text-primary;
  background: $color-bg-secondary;
}
</style>
