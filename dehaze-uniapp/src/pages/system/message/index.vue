<template>
  <PageLayout level="L2" title="消息管理">
    <view class="page-body">
      <view class="tabs">
        <view
          :class="['tab', { active: currentTab === 0 }]"
          @click="currentTab = 0"
        >
          <text>公告</text>
        </view>
        <view
          :class="['tab', { active: currentTab === 1 }]"
          @click="currentTab = 1"
        >
          <text>模板</text>
        </view>
      </view>

      <!-- 公告管理 -->
      <view v-if="currentTab === 0">
        <view class="search-bar">
          <view class="search-row">
            <input
              class="search-input"
              placeholder="搜索公告标题"
              v-model="annKeyword"
              confirm-type="search"
              @confirm="handleAnnSearch"
            />
            <button
              v-if="canAddAnnouncement"
              class="btn btn-primary btn-sm"
              @click="openCreateAnn"
            >
              新建
            </button>
          </view>
          <view class="status-filter-row">
            <view
              v-for="s in statusFilters"
              :key="s.label"
              class="status-filter-item"
              :class="{ active: annStatusFilter === s.value }"
              @click="
                () => {
                  annStatusFilter = s.value;
                  fetchAnnouncements(true);
                }
              "
            >
              <text>{{ s.label }}</text>
            </view>
          </view>
        </view>

        <view
          v-if="annLoading && announcements.length === 0"
          class="loading-container"
        >
          <view class="loading-spinner" />
          <text class="loading-text">加载中...</text>
        </view>
        <view v-else-if="announcements.length === 0" class="empty-tip"
          >暂无公告</view
        >
        <view v-else class="list">
          <view v-for="a in announcements" :key="a.id" class="list-card">
            <view class="card-header">
              <view class="card-title-row">
                <text class="card-name">{{ a.title }}</text>
                <text class="status-tag" :class="'status-' + a.status">
                  {{ a.statusLabel || getStatusLabel(a.status) }}
                </text>
              </view>
              <text class="card-type">{{ a.typeLabel || a.type }}</text>
            </view>
            <view class="card-meta">
              <text class="meta-item"
                >目标: {{ a.targetScopeLabel || a.targetScope }}</text
              >
              <text class="meta-item"
                >重要度: {{ a.importanceLabel || a.importance }}</text
              >
              <text v-if="a.sentCount !== undefined" class="meta-item"
                >发送: {{ a.sentCount }}人</text
              >
            </view>
            <view class="card-meta">
              <text class="meta-item">{{ formatDateTime(a.createTime) }}</text>
            </view>
            <view class="card-actions">
              <button
                v-if="(a.status === 1 || a.status === 2) && canEditAnnouncement"
                class="btn btn-sm action-btn"
                @click="openEditAnn(a)"
              >
                编辑
              </button>
              <button
                v-if="(a.status === 1 || a.status === 2) && canSendAnnouncement"
                class="btn btn-sm action-btn btn-success"
                @click="handleSendAnnouncement(a.id)"
              >
                发送
              </button>
              <button
                v-if="a.status === 2 && canCancelAnnouncement"
                class="btn btn-sm action-btn btn-warning"
                @click="handleCancelAnnouncement(a.id)"
              >
                取消
              </button>
              <button
                v-if="canDeleteAnnouncement"
                class="btn btn-sm action-btn btn-danger"
                @click="handleDeleteAnnouncement(a.id)"
              >
                删除
              </button>
            </view>
          </view>
          <view v-if="hasMoreAnn" class="load-more" @click="handleLoadMoreAnn">
            <text>加载更多</text>
          </view>
          <view v-else-if="announcements.length > 0" class="end-text"
            >— 没有更多了 —</view
          >
        </view>
      </view>

      <!-- 模板管理 -->
      <view v-else>
        <view
          v-if="tplLoading && templates.length === 0"
          class="loading-container"
        >
          <view class="loading-spinner" />
          <text class="loading-text">加载中...</text>
        </view>
        <view v-else-if="templates.length === 0" class="empty-tip"
          >暂无模板</view
        >
        <view v-else class="list">
          <view v-for="t in templates" :key="t.id" class="list-card">
            <view class="card-header">
              <view class="card-title-row">
                <text class="card-name">{{ t.name }}</text>
                <text
                  class="status-tag"
                  :class="t.status === 1 ? 'status-3' : 'status-1'"
                >
                  {{ t.status === 1 ? "启用" : "禁用" }}
                </text>
              </view>
              <text class="card-id">{{ t.code }}</text>
            </view>
            <view class="card-meta">
              <text class="meta-item">类型: {{ t.type }}</text>
              <text class="meta-item">优先级: {{ t.priority }}</text>
            </view>
            <text class="card-content">标题模板: {{ t.titleTemplate }}</text>
            <view
              v-if="t.variables && t.variables.length > 0"
              class="card-meta"
            >
              <text class="meta-item"
                >变量:
                {{
                  t.variables.map((v: any) => "{" + v.name + "}").join(" ")
                }}</text
              >
            </view>
            <view class="card-actions">
              <button
                v-if="canManageTemplate"
                class="btn btn-sm action-btn"
                @click="openEditTpl(t)"
              >
                编辑
              </button>
            </view>
          </view>
          <view v-if="hasMoreTpl" class="load-more" @click="handleLoadMoreTpl">
            <text>加载更多</text>
          </view>
          <view v-else-if="templates.length > 0" class="end-text"
            >— 没有更多了 —</view
          >
        </view>
      </view>

      <!-- 公告编辑弹窗 -->
      <Popup
        :show="showAnnouncementForm"
        mode="bottom"
        round
        @close="showAnnouncementForm = false"
      >
        <view class="popup-content">
          <text class="popup-title">{{
            editingAnn ? "编辑公告" : "新建公告"
          }}</text>
          <view class="form-item">
            <text class="form-label">标题 *</text>
            <input
              class="form-input"
              v-model="annForm.title"
              placeholder="公告标题"
            />
          </view>
          <view class="form-item">
            <text class="form-label">内容</text>
            <textarea
              class="form-textarea"
              v-model="annForm.content"
              placeholder="公告内容"
            />
          </view>
          <view class="popup-footer">
            <button
              class="btn btn-default"
              @click="showAnnouncementForm = false"
            >
              取消
            </button>
            <button
              class="btn btn-primary"
              :disabled="savingAnn"
              @click="handleSaveAnn"
            >
              保存
            </button>
          </view>
        </view>
      </Popup>

      <!-- 模板编辑弹窗 -->
      <Popup
        :show="showTemplateForm"
        mode="bottom"
        round
        @close="showTemplateForm = false"
      >
        <view class="popup-content">
          <text class="popup-title">编辑模板</text>
          <view class="form-item">
            <text class="form-label">模板名称</text>
            <input
              class="form-input"
              v-model="tplForm.name"
              placeholder="模板名称"
            />
          </view>
          <view class="form-item">
            <text class="form-label">标题模板</text>
            <input
              class="form-input"
              v-model="tplForm.titleTemplate"
              placeholder="标题模板"
            />
          </view>
          <view class="form-item">
            <text class="form-label">内容模板</text>
            <textarea
              class="form-textarea"
              v-model="tplForm.contentTemplate"
              placeholder="内容模板"
            />
          </view>
          <view class="popup-footer">
            <button class="btn btn-default" @click="showTemplateForm = false">
              取消
            </button>
            <button
              class="btn btn-primary"
              :disabled="savingTpl"
              @click="handleSaveTpl"
            >
              保存
            </button>
          </view>
        </view>
      </Popup>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import { usePagedList } from "@/composables/usePagedList";
import { AnnouncementAPI, MessageTemplateAPI } from "dehaze-sdk-js";
import type {
  AnnouncementVO,
  AnnouncementForm,
  MessageTemplateVO,
} from "dehaze-sdk-js";

const STATUS_LABEL: Record<number, string> = {
  1: "草稿",
  2: "待发送",
  3: "已发送",
  4: "已取消",
};
const statusFilters = [
  { label: "全部", value: undefined },
  { label: "草稿", value: 1 },
  { label: "待发送", value: 2 },
  { label: "已发送", value: 3 },
  { label: "已取消", value: 4 },
];

const currentTab = ref(0);

// 权限（管理端默认有权限，message 模块不做权限控制）
const canAddAnnouncement = ref(true);
const canEditAnnouncement = ref(true);
const canDeleteAnnouncement = ref(true);
const canSendAnnouncement = ref(true);
const canCancelAnnouncement = ref(true);
const canManageTemplate = ref(true);

const annStatusFilter = ref<number | undefined>();

const {
  list: announcements,
  keyword: annKeyword,
  hasMore: hasMoreAnn,
  loading: annLoading,
  fetchList: fetchAnnouncements,
  handleSearch: handleAnnSearch,
  loadMore: handleLoadMoreAnn,
} = usePagedList<AnnouncementVO>({
  fetcher: (p) => {
    const params: any = { pageNum: p.pageNum, pageSize: 15 };
    if (p.keyword) params.title = p.keyword;
    if (annStatusFilter.value !== undefined)
      params.status = annStatusFilter.value;
    return AnnouncementAPI.getPage(params).then((r) => r.list || []);
  },
});

const {
  list: templates,
  hasMore: hasMoreTpl,
  loading: tplLoading,
  fetchList: fetchTemplates,
  loadMore: handleLoadMoreTpl,
} = usePagedList<MessageTemplateVO>({
  fetcher: (p) =>
    MessageTemplateAPI.getPage({ pageNum: p.pageNum, pageSize: 15 }).then(
      (r) => r.list || []
    ),
});

const showAnnouncementForm = ref(false);
const showTemplateForm = ref(false);
const editingAnn = ref<AnnouncementVO | null>(null);
const editingTpl = ref<MessageTemplateVO | null>(null);
const savingAnn = ref(false);
const savingTpl = ref(false);

const annForm = ref({
  title: "",
  content: "",
  type: "operation",
  importance: 1,
  targetScope: "all",
  sendTime: "",
  expireTime: "",
});

const tplForm = ref({
  name: "",
  titleTemplate: "",
  contentTemplate: "",
  priority: 1,
  status: 1,
});

function formatDateTime(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function getStatusLabel(status: number): string {
  return STATUS_LABEL[status] || `未知(${status})`;
}

const handleSendAnnouncement = async (id: number) => {
  try {
    const result = await AnnouncementAPI.send(id);
    uni.showToast({
      title: `已发送给 ${result.sentCount} 人`,
      icon: "success",
    });
    fetchAnnouncements(true);
  } catch {
    uni.showToast({ title: "发送失败", icon: "error" });
  }
};

const handleCancelAnnouncement = async (id: number) => {
  try {
    await AnnouncementAPI.cancel(id);
    uni.showToast({ title: "已取消", icon: "success" });
    fetchAnnouncements(true);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};

const handleDeleteAnnouncement = async (id: number) => {
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定要删除这条公告吗？",
  });
  if (!res.confirm) return;
  try {
    await AnnouncementAPI.deleteById(id);
    uni.showToast({ title: "已删除", icon: "success" });
    fetchAnnouncements(true);
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};

const openCreateAnn = () => {
  editingAnn.value = null;
  annForm.value = {
    title: "",
    content: "",
    type: "operation",
    importance: 1,
    targetScope: "all",
    sendTime: "",
    expireTime: "",
  };
  showAnnouncementForm.value = true;
};

const openEditAnn = (a: AnnouncementVO) => {
  if (a.status !== 1 && a.status !== 2) {
    uni.showToast({ title: "仅草稿/待发送可编辑", icon: "none" });
    return;
  }
  editingAnn.value = a;
  annForm.value = {
    title: a.title,
    content: a.content || "",
    type: a.type,
    importance: a.importance,
    targetScope: a.targetScope,
    sendTime: a.sendTime || "",
    expireTime: a.expireTime || "",
  };
  showAnnouncementForm.value = true;
};

const handleSaveAnn = async () => {
  if (!annForm.value.title.trim()) {
    uni.showToast({ title: "请输入公告标题", icon: "none" });
    return;
  }
  savingAnn.value = true;
  try {
    const data: AnnouncementForm = {
      title: annForm.value.title,
      content: annForm.value.content,
      type: annForm.value.type,
      importance: annForm.value.importance,
      targetScope: annForm.value.targetScope,
    };
    if (annForm.value.sendTime) data.sendTime = annForm.value.sendTime;
    if (annForm.value.expireTime) data.expireTime = annForm.value.expireTime;

    if (editingAnn.value) {
      await AnnouncementAPI.update(editingAnn.value.id, data);
      uni.showToast({ title: "更新成功", icon: "success" });
    } else {
      await AnnouncementAPI.create(data);
      uni.showToast({ title: "创建成功", icon: "success" });
    }
    showAnnouncementForm.value = false;
    fetchAnnouncements(true);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  } finally {
    savingAnn.value = false;
  }
};

const openEditTpl = (tpl: MessageTemplateVO) => {
  editingTpl.value = tpl;
  tplForm.value = {
    name: tpl.name,
    titleTemplate: tpl.titleTemplate,
    contentTemplate: tpl.contentTemplate || "",
    priority: tpl.priority,
    status: tpl.status,
  };
  showTemplateForm.value = true;
};

const handleSaveTpl = async () => {
  if (!editingTpl.value) return;
  savingTpl.value = true;
  try {
    await MessageTemplateAPI.update(editingTpl.value.id, {
      name: tplForm.value.name,
      titleTemplate: tplForm.value.titleTemplate,
      contentTemplate: tplForm.value.contentTemplate,
      priority: tplForm.value.priority,
      status: tplForm.value.status,
    });
    uni.showToast({ title: "保存成功", icon: "success" });
    showTemplateForm.value = false;
    fetchTemplates(true);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    savingTpl.value = false;
  }
};

fetchAnnouncements(true);
fetchTemplates(true);
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
  padding-bottom: 60rpx;
}

.tabs {
  display: flex;
  background: $color-white;
  border-radius: $radius-lg;
  margin-bottom: $spacing-md;
  overflow: hidden;
}
.tab {
  flex: 1;
  text-align: center;
  padding: 24rpx;
  font-size: $font-md;
  color: $color-text-secondary;
  &.active {
    color: $color-primary;
    font-weight: 600;
    background: $color-primary-bg;
  }
}

.search-bar {
  background: $color-white;
  border-radius: $radius-lg;
  padding: 16rpx;
  margin-bottom: $spacing-md;
}
.search-row {
  display: flex;
  gap: 16rpx;
  align-items: center;
}
.search-input {
  flex: 1;
  padding: 14rpx 20rpx;
  font-size: $font-md;
  background: $color-bg-secondary;
  border-radius: $radius-md;
}
.status-filter-row {
  display: flex;
  gap: 12rpx;
  margin-top: 16rpx;
  flex-wrap: wrap;
}
.status-filter-item {
  padding: 8rpx 20rpx;
  font-size: $font-xs;
  color: $color-text-secondary;
  background: $color-bg-secondary;
  border-radius: 24rpx;
  &.active {
    color: $color-white;
    background: $color-primary;
  }
}

.list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.list-card {
  padding: 24rpx;
  background: $color-white;
  border-radius: $radius-lg;
  box-shadow: $shadow-sm;
}
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16rpx;
}
.card-title-row {
  display: flex;
  flex: 1;
  gap: 12rpx;
  align-items: center;
  min-width: 0;
}
.card-name {
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
  white-space: nowrap;
}
.card-id {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.card-type {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx 24rpx;
  margin-bottom: 12rpx;
}
.meta-item {
  font-size: $font-xs;
  color: $color-text-secondary;
}
.card-content {
  display: block;
  margin-bottom: 12rpx;
  font-size: $font-sm;
  line-height: 1.6;
  color: $color-text-primary;
}
.card-actions {
  display: flex;
  gap: 16rpx;
  justify-content: flex-end;
}
.action-btn {
  color: $color-primary;
  background: $color-primary-bg;
  &.btn-success {
    color: $color-white;
    background: $color-success;
  }
  &.btn-warning {
    color: $color-white;
    background: $color-warning;
  }
  &.btn-danger {
    color: $color-white;
    background: $color-danger;
  }
}
.status-tag {
  padding: 4rpx 12rpx;
  font-size: $font-xs;
  border-radius: 8rpx;
  &.status-1 {
    color: $color-text-secondary;
    background: $color-bg-secondary;
  }
  &.status-2 {
    color: $color-warning;
    background: $color-warning-bg;
  }
  &.status-3 {
    color: $color-success;
    background: $color-success-bg;
  }
  &.status-4 {
    color: $color-text-secondary;
    background: $color-bg-secondary;
  }
}

.load-more {
  text-align: center;
  font-size: $font-sm;
  color: $color-secondary;
  padding: 24rpx 0;
}
.end-text {
  text-align: center;
  font-size: $font-sm;
  color: $color-text-disabled;
  padding: 32rpx 0;
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
  margin-bottom: 16rpx;
}
.form-label {
  display: block;
  margin-bottom: 8rpx;
  font-size: $font-sm;
  font-weight: 500;
  color: $color-text-primary;
}
.form-input {
  box-sizing: border-box;
  width: 100%;
  padding: 16rpx 20rpx;
  font-size: $font-md;
  border: 1rpx solid $color-border;
  border-radius: $radius-md;
}
.form-textarea {
  box-sizing: border-box;
  width: 100%;
  min-height: 160rpx;
  padding: 16rpx 20rpx;
  font-size: $font-md;
  border: 1rpx solid $color-border;
  border-radius: $radius-md;
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
.btn-sm {
  padding: 6rpx 16rpx;
  font-size: $font-xs;
}
.btn-primary {
  color: $color-white;
  background: $color-primary;
}
.btn-success {
  color: $color-white;
  background: $color-success;
}
.btn-warning {
  color: $color-white;
  background: $color-warning;
}
.btn-danger {
  color: $color-white;
  background: $color-danger;
}
.btn-default {
  color: $color-text-primary;
  background: $color-bg-secondary;
}
</style>
