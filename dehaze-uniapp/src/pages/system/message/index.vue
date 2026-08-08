<template>
  <PageLayout level="L2" title="消息管理">
    <view class="page-body">
      <u-tabs
        :list="tabs"
        :current="currentTab"
        @change="(i: any) => (currentTab = i.index)"
      />
      <!-- 公告管理 -->
      <view v-if="currentTab === 0">
        <view class="search-bar">
          <view class="search-row">
            <input
              class="search-input"
              placeholder="搜索公告标题"
              v-model="annKeyword"
              @confirm="handleAnnSearch"
            />
            <view v-if="canAddAnnouncement" class="create-btn" @click="openCreateAnn">
              <text>新建</text>
            </view>
          </view>
          <view class="status-filter-row">
            <view
              v-for="s in statusFilters"
              :key="s.value"
              class="status-filter-item"
              :class="{ active: annStatusFilter === s.value }"
              @click="() => { annStatusFilter = s.value; fetchAnnouncements(1, annKeyword, s.value); }"
            >
              <text>{{ s.label }}</text>
            </view>
          </view>
        </view>
        <scroll-view scroll-y class="list-scroll" @scrolltolower="handleLoadMoreAnn">
          <view v-if="annLoading && announcements.length === 0" class="loading-wrapper">
            <text>加载中...</text>
          </view>
          <view v-else-if="announcements.length === 0" class="empty-wrapper">
            <text>暂无公告</text>
          </view>
          <view v-else>
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
                <text class="meta-item">目标: {{ a.targetScopeLabel || a.targetScope }}</text>
                <text class="meta-item">重要度: {{ a.importanceLabel || a.importance }}</text>
                <text v-if="a.sentCount !== undefined" class="meta-item">发送: {{ a.sentCount }}人</text>
              </view>
              <view class="card-meta">
                <text class="meta-item">{{ formatDateTime(a.createTime) }}</text>
              </view>
              <view class="card-actions">
                <view
                  v-if="(a.status === 1 || a.status === 2) && canEditAnnouncement"
                  class="action-btn"
                  @click="openEditAnn(a)"
                >编辑</view>
                <view
                  v-if="(a.status === 1 || a.status === 2) && canSendAnnouncement"
                  class="action-btn primary"
                  @click="handleSendAnnouncement(a.id)"
                >发送</view>
                <view
                  v-if="a.status === 2 && canCancelAnnouncement"
                  class="action-btn warning"
                  @click="handleCancelAnnouncement(a.id)"
                >取消</view>
                <view
                  v-if="canDeleteAnnouncement"
                  class="action-btn danger"
                  @click="handleDeleteAnnouncement(a.id)"
                >删除</view>
              </view>
            </view>
            <view v-if="announcements.length > 0 && announcements.length < annTotal" class="load-more" @click="handleLoadMoreAnn">
              <text>加载更多</text>
            </view>
          </view>
        </scroll-view>
      </view>

      <!-- 模板管理 -->
      <view v-if="currentTab === 1">
        <scroll-view scroll-y class="list-scroll" @scrolltolower="handleLoadMoreTpl">
          <view v-if="tplLoading && templates.length === 0" class="loading-wrapper">
            <text>加载中...</text>
          </view>
          <view v-else-if="templates.length === 0" class="empty-wrapper">
            <text>暂无模板</text>
          </view>
          <view v-else>
            <view v-for="t in templates" :key="t.id" class="list-card">
              <view class="card-header">
                <view class="card-title-row">
                  <text class="card-name">{{ t.name }}</text>
                  <text class="status-tag" :class="t.status === 1 ? 'status-3' : 'status-1'">
                    {{ t.status === 1 ? "启用" : "禁用" }}
                  </text>
                </view>
                <text class="card-id">{{ t.code }}</text>
              </view>
              <view class="card-meta">
                <text class="meta-item">类型: {{ t.type }}</text>
                <text class="meta-item">优先级: {{ t.priority }}</text>
              </view>
              <text class="card-content" numberOfLines="1">标题模板: {{ t.titleTemplate }}</text>
              <view v-if="t.variables && t.variables.length > 0" class="card-meta">
                <text class="meta-item">变量: {{ t.variables.map((v: any) => '{' + v.name + '}').join(" ") }}</text>
              </view>
              <view class="card-actions">
                <view v-if="canManageTemplate" class="action-btn" @click="openEditTpl(t)">编辑</view>
              </view>
            </view>
            <view v-if="templates.length > 0 && templates.length < tplTotal" class="load-more" @click="handleLoadMoreTpl">
              <text>加载更多</text>
            </view>
          </view>
        </scroll-view>
      </view>

      <!-- 公告编辑弹窗 -->
      <u-popup :show="showAnnouncementForm" @close="showAnnouncementForm = false" round>
        <view class="popup-content">
          <view class="popup-title">{{ editingAnn ? "编辑公告" : "新建公告" }}</view>
          <view class="popup-body">
            <view class="form-item">
              <text class="form-label">标题 *</text>
              <input class="form-input" v-model="annForm.title" placeholder="公告标题" />
            </view>
            <view class="form-item">
              <text class="form-label">内容</text>
              <textarea class="form-textarea" v-model="annForm.content" placeholder="公告内容" />
            </view>
            <u-button type="primary" @click="handleSaveAnn" :loading="savingAnn">保存</u-button>
          </view>
        </view>
      </u-popup>

      <!-- 模板编辑弹窗 -->
      <u-popup :show="showTemplateForm" @close="showTemplateForm = false" round>
        <view class="popup-content">
          <view class="popup-title">编辑模板</view>
          <view class="popup-body">
            <view class="form-item">
              <text class="form-label">模板名称</text>
              <input class="form-input" v-model="tplForm.name" placeholder="模板名称" />
            </view>
            <view class="form-item">
              <text class="form-label">标题模板</text>
              <input class="form-input" v-model="tplForm.titleTemplate" placeholder="标题模板" />
            </view>
            <view class="form-item">
              <text class="form-label">内容模板</text>
              <textarea class="form-textarea" v-model="tplForm.contentTemplate" placeholder="内容模板" />
            </view>
            <u-button type="primary" @click="handleSaveTpl" :loading="savingTpl">保存</u-button>
          </view>
        </view>
      </u-popup>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { AnnouncementAPI, MessageTemplateAPI } from "dehaze-sdk-js";
import type { AnnouncementVO, AnnouncementForm, MessageTemplateVO } from "dehaze-sdk-js";

const STATUS_LABEL: Record<number, string> = { 1: "草稿", 2: "待发送", 3: "已发送", 4: "已取消" };
const statusFilters = [
  { label: "全部", value: undefined },
  { label: "草稿", value: 1 },
  { label: "待发送", value: 2 },
  { label: "已发送", value: 3 },
  { label: "已取消", value: 4 },
];

const tabs = [{ name: "公告" }, { name: "模板" }];
const currentTab = ref(0);

// 权限（简化处理，管理端默认有权限）
const canAddAnnouncement = ref(true);
const canEditAnnouncement = ref(true);
const canDeleteAnnouncement = ref(true);
const canSendAnnouncement = ref(true);
const canCancelAnnouncement = ref(true);
const canManageTemplate = ref(true);

const announcements = ref<AnnouncementVO[]>([]);
const annLoading = ref(false);
const annTotal = ref(0);
const annPageNum = ref(1);
const annKeyword = ref("");
const annStatusFilter = ref<number | undefined>();

const templates = ref<MessageTemplateVO[]>([]);
const tplLoading = ref(false);
const tplTotal = ref(0);
const tplPageNum = ref(1);

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

const fetchAnnouncements = async (page: number, kw: string, status?: number) => {
  annLoading.value = true;
  try {
    const params: any = { pageNum: page, pageSize: 15 };
    if (kw) params.title = kw;
    if (status !== undefined) params.status = status;
    const res = await AnnouncementAPI.getPage(params);
    announcements.value = res.list;
    annTotal.value = res.total;
    annPageNum.value = page;
  } catch {
    uni.showToast({ title: "加载公告失败", icon: "none" });
  } finally {
    annLoading.value = false;
  }
};

const fetchTemplates = async (page: number) => {
  tplLoading.value = true;
  try {
    const res = await MessageTemplateAPI.getPage({ pageNum: page, pageSize: 15 });
    templates.value = res.list;
    tplTotal.value = res.total;
    tplPageNum.value = page;
  } catch {
    uni.showToast({ title: "加载模板失败", icon: "none" });
  } finally {
    tplLoading.value = false;
  }
};

const handleAnnSearch = () => {
  fetchAnnouncements(1, annKeyword.value, annStatusFilter.value);
};

const handleLoadMoreAnn = () => {
  if (announcements.value.length < annTotal.value) {
    fetchAnnouncements(annPageNum.value + 1, annKeyword.value, annStatusFilter.value);
  }
};

const handleLoadMoreTpl = () => {
  if (templates.value.length < tplTotal.value) {
    fetchTemplates(tplPageNum.value + 1);
  }
};

const handleSendAnnouncement = async (id: number) => {
  try {
    const result = await AnnouncementAPI.send(id);
    uni.showToast({ title: `已发送给 ${result.sentCount} 人`, icon: "success" });
    fetchAnnouncements(annPageNum.value, annKeyword.value, annStatusFilter.value);
  } catch {
    uni.showToast({ title: "发送失败", icon: "error" });
  }
};

const handleCancelAnnouncement = async (id: number) => {
  try {
    await AnnouncementAPI.cancel(id);
    uni.showToast({ title: "已取消", icon: "success" });
    fetchAnnouncements(annPageNum.value, annKeyword.value, annStatusFilter.value);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};

const handleDeleteAnnouncement = async (id: number) => {
  const res = await uni.showModal({ title: "确认删除", content: "确定要删除这条公告吗？" });
  if (!res.confirm) return;
  try {
    await AnnouncementAPI.deleteById(id);
    uni.showToast({ title: "已删除", icon: "success" });
    fetchAnnouncements(annPageNum.value, annKeyword.value, annStatusFilter.value);
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};

const openCreateAnn = () => {
  editingAnn.value = null;
  annForm.value = { title: "", content: "", type: "operation", importance: 1, targetScope: "all", sendTime: "", expireTime: "" };
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
    fetchAnnouncements(1, annKeyword.value, annStatusFilter.value);
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
    fetchTemplates(tplPageNum.value);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    savingTpl.value = false;
  }
};

onMounted(() => {
  fetchAnnouncements(1, "", undefined);
  fetchTemplates(1);
});
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
  min-height: 100vh;
  background: $color-bg-primary;
}

.search-bar {
  padding: 20rpx 16rpx;
}
.search-row {
  display: flex;
  gap: 16rpx;
  align-items: center;
}
.search-input {
  flex: 1;
  padding: 14rpx 20rpx;
  font-size: 28rpx;
  background: #f3f4f6;
  border-radius: 16rpx;
}
.create-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14rpx 28rpx;
  font-size: 26rpx;
  font-weight: 500;
  color: #fff;
  white-space: nowrap;
  background: #3b82f6;
  border-radius: 16rpx;
}
.status-filter-row {
  display: flex;
  gap: 12rpx;
  margin-top: 16rpx;
}
.status-filter-item {
  padding: 8rpx 20rpx;
  font-size: 24rpx;
  color: #6b7280;
  background: #f3f4f6;
  border-radius: 24rpx;
  &.active {
    color: #fff;
    background: #3b82f6;
  }
}

.list-scroll {
  flex: 1;
  padding: 0 16rpx;
}
.loading-wrapper,
.empty-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 160rpx 0;
  font-size: 28rpx;
  color: #9ca3af;
}

.list-card {
  padding: 24rpx;
  margin-bottom: 16rpx;
  background: #fff;
  border-radius: 16rpx;
  box-shadow: 0 2rpx 4rpx rgb(0 0 0 / 4%);
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
  font-size: 30rpx;
  font-weight: 600;
  color: #1f2937;
  white-space: nowrap;
}
.card-id {
  font-size: 22rpx;
  color: #9ca3af;
}
.card-type {
  font-size: 22rpx;
  color: #9ca3af;
}
.card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx 24rpx;
  margin-bottom: 12rpx;
}
.meta-item {
  font-size: 24rpx;
  color: #6b7280;
}
.card-content {
  display: block;
  margin-bottom: 12rpx;
  font-size: 26rpx;
  line-height: 1.6;
  color: #374151;
}
.card-actions {
  display: flex;
  gap: 16rpx;
  justify-content: flex-end;
}
.action-btn {
  padding: 10rpx 20rpx;
  font-size: 24rpx;
  color: #3b82f6;
  background: #eff6ff;
  border-radius: 12rpx;
  &.danger {
    color: #ef4444;
    background: #fef2f2;
  }
  &.primary {
    color: #10b981;
    background: #ecfdf5;
  }
  &.warning {
    color: #f59e0b;
    background: #fffbeb;
  }
}
.status-tag {
  padding: 4rpx 12rpx;
  font-size: 22rpx;
  border-radius: 8rpx;
  &.status-1 { color: #6b7280; background: #f3f4f6; }
  &.status-2 { color: #d97706; background: #fef3c7; }
  &.status-3 { color: #059669; background: #d1fae5; }
  &.status-4 { color: #6b7280; background: #f3f4f6; }
}
.load-more {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24rpx;
  font-size: 26rpx;
  color: #3b82f6;
}

/* 弹窗 */
.popup-content {
  display: flex;
  flex-direction: column;
  max-height: 75vh;
  padding: 32rpx;
  background: #fff;
  border-radius: 24rpx 24rpx 0 0;
}
.popup-title {
  font-size: 32rpx;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 24rpx;
}
.popup-body {
  display: flex;
  flex-direction: column;
  gap: 24rpx;
}
.form-item {
  margin-bottom: 16rpx;
}
.form-label {
  display: block;
  margin-bottom: 8rpx;
  font-size: 26rpx;
  font-weight: 500;
  color: #374151;
}
.form-input {
  box-sizing: border-box;
  width: 100%;
  padding: 16rpx 20rpx;
  font-size: 28rpx;
  border: 2rpx solid #d1d5db;
  border-radius: 12rpx;
}
.form-textarea {
  box-sizing: border-box;
  width: 100%;
  height: 160rpx;
  padding: 16rpx 20rpx;
  font-size: 28rpx;
  border: 2rpx solid #d1d5db;
  border-radius: 12rpx;
}
</style>
