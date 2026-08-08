<template>
  <PageLayout level="L2" title="算法管理">
    <view class="page-body">
      <!-- 搜索栏 + 新增按钮 -->
      <view class="search-bar">
        <u-search
          v-model="keyword"
          placeholder="搜索算法名称或类型"
          @search="handleSearch"
          @clear="handleSearch"
        />
        <u-button
          v-if="canAdd"
          type="primary"
          size="small"
          class="add-btn"
          @click="openAdd"
        >
          新增
        </u-button>
      </view>

      <!-- 状态筛选 -->
      <view class="filter-row">
        <u-tag
          v-for="f in statusFilters"
          :key="f.value"
          :text="f.label"
          :type="statusFilter === f.value ? 'primary' : 'info'"
          @click="statusFilter = f.value"
        />
      </view>

      <!-- 算法列表（树形） -->
      <view v-if="loading" class="loading-wrapper">
        <u-loading-icon text="加载中..." />
      </view>
      <view v-else-if="flatList.length === 0" class="empty-wrapper">
        <u-empty text="暂无算法数据" />
      </view>
      <view v-else class="algo-list">
        <view
          v-for="item in flatList"
          :key="item.algorithm.id"
          class="algo-node"
          :style="{ marginLeft: item.level * 24 + 'px' }"
          @click="handleDetail(item.algorithm)"
        >
          <view class="node-main">
            <view class="node-info">
              <text class="node-icon">{{ item.hasChildren ? '📁' : '📄' }}</text>
              <text class="node-name">{{ item.algorithm.name }}</text>
            </view>
            <view class="node-meta">
              <u-tag
                :text="statusLabel(item.algorithm.status)"
                :type="statusTagType(item.algorithm.status)"
                size="mini"
              />
              <text v-if="item.algorithm.type" class="node-type">{{ item.algorithm.type }}</text>
              <text v-if="item.algorithm.version" class="node-version">v{{ item.algorithm.version }}</text>
            </view>
          </view>

          <!-- 操作按钮 -->
          <view
            v-if="!item.hasChildren && (canAudit || canEdit || canDelete)"
            class="node-actions"
            @click.stop
          >
            <u-button
              v-if="item.algorithm.status === 3 && canAudit"
              type="success"
              size="small"
              @click="openAudit(item.algorithm, true)"
            >
              通过
            </u-button>
            <u-button
              v-if="item.algorithm.status === 3 && canAudit"
              type="error"
              size="small"
              @click="openAudit(item.algorithm, false)"
            >
              驳回
            </u-button>
            <u-button
              v-if="(item.algorithm.status === 4 || item.algorithm.status === 5) && canEdit"
              :type="item.algorithm.status === 4 ? 'warning' : 'primary'"
              size="small"
              :loading="actionLoadingId === item.algorithm.id"
              @click="handleToggleStatus(item.algorithm)"
            >
              {{ item.algorithm.status === 4 ? '停用' : '启用' }}
            </u-button>
            <u-button
              v-if="isDeletable(item.algorithm.status) && canDelete"
              type="error"
              size="small"
              :loading="actionLoadingId === item.algorithm.id"
              @click="handleDelete(item.algorithm)"
            >
              删除
            </u-button>
          </view>
        </view>
      </view>

      <!-- 算法详情弹窗 -->
      <u-popup
        :show="detailVisible"
        mode="bottom"
        round="20"
        @close="detailVisible = false"
      >
        <scroll-view scroll-y class="detail-content">
          <view class="detail-header">
            <text class="detail-title">{{ detailAlgo?.name }}</text>
            <text class="detail-close" @click="detailVisible = false">关闭</text>
          </view>

          <view class="detail-section">
            <text class="section-title">基本信息</text>
            <DetailItem label="算法名称" :value="detailAlgo?.name" />
            <DetailItem label="算法类型" :value="detailAlgo?.type" />
            <DetailItem label="描述" :value="detailAlgo?.description" />
            <DetailItem label="状态">
              <u-tag
                :text="statusLabel(detailAlgo?.status)"
                :type="statusTagType(detailAlgo?.status)"
                size="mini"
              />
            </DetailItem>
            <DetailItem label="版本" :value="detailAlgo?.version" />
            <DetailItem label="大小" :value="detailAlgo?.size" />
          </view>

          <view class="detail-section">
            <text class="section-title">技术信息</text>
            <DetailItem label="路径" :value="detailAlgo?.path" />
            <DetailItem label="导入路径" :value="detailAlgo?.importPath" />
            <DetailItem label="参数" :value="detailAlgo?.params" />
            <DetailItem label="计算量(FLOPs)" :value="detailAlgo?.flops" />
          </view>

          <view class="detail-section" v-if="detailAlgo?.auditBy != null">
            <text class="section-title">审核信息</text>
            <DetailItem label="审核人" :value="String(detailAlgo?.auditBy || '')" />
            <DetailItem label="审核时间" :value="detailAlgo?.auditTime" />
            <DetailItem label="审核备注" :value="detailAlgo?.auditRemark" />
          </view>

          <DetailItem label="创建时间" :value="detailAlgo?.createTime" />

          <view class="detail-footer">
            <u-button
              v-if="detailAlgo?.status === 3 && canAudit"
              type="success"
              @click="openAudit(detailAlgo!, true)"
            >
              审核通过
            </u-button>
            <u-button
              v-if="detailAlgo?.status === 3 && canAudit"
              type="error"
              @click="openAudit(detailAlgo!, false)"
            >
              审核驳回
            </u-button>
            <u-button
              v-if="detailAlgo?.status === 4 && canEdit"
              type="warning"
              :loading="actionLoadingId === detailAlgo?.id"
              @click="handleToggleStatus(detailAlgo!)"
            >
              停用算法
            </u-button>
            <u-button
              v-if="detailAlgo?.status === 5 && canEdit"
              type="primary"
              :loading="actionLoadingId === detailAlgo?.id"
              @click="handleToggleStatus(detailAlgo!)"
            >
              启用算法
            </u-button>
            <u-button
              v-if="isDeletable(detailAlgo?.status) && canDelete"
              type="error"
              :loading="actionLoadingId === detailAlgo?.id"
              @click="handleDelete(detailAlgo!)"
            >
              删除算法
            </u-button>
          </view>
        </scroll-view>
      </u-popup>

      <!-- 审核弹窗 -->
      <u-popup
        :show="auditVisible"
        mode="center"
        round="20"
        @close="auditVisible = false"
      >
        <view class="audit-content">
          <text class="audit-title">{{ auditApproved ? '审核通过' : '审核驳回' }}</text>
          <text v-if="auditAlgo" class="audit-name">算法：{{ auditAlgo.name }}</text>
          <view v-if="!auditApproved" class="audit-remark">
            <text class="remark-label">驳回原因（必填）</text>
            <u-textarea
              v-model="auditRemark"
              placeholder="请输入驳回原因"
              maxlength="200"
            />
          </view>
          <view class="audit-footer">
            <u-button @click="auditVisible = false">取消</u-button>
            <u-button
              :type="auditApproved ? 'success' : 'error'"
              :loading="auditSubmitting"
              @click="handleAuditSubmit"
            >
              确认
            </u-button>
          </view>
        </view>
      </u-popup>

      <!-- 新增算法弹窗 -->
      <u-popup
        :show="addVisible"
        mode="bottom"
        round="20"
        @close="addVisible = false"
      >
        <scroll-view scroll-y class="detail-content">
          <view class="detail-header">
            <text class="detail-title">新增算法</text>
            <text class="detail-close" @click="addVisible = false">关闭</text>
          </view>
          <view class="detail-section">
            <text class="section-title">基本信息</text>
            <view class="form-item">
              <text class="form-label">名称 *</text>
              <u-input v-model="addForm.name" placeholder="算法名称" border="bottom" />
            </view>
            <view class="form-item">
              <text class="form-label">类型 *</text>
              <u-input v-model="addForm.type" placeholder="算法类型" border="bottom" />
            </view>
            <view class="form-item">
              <text class="form-label">版本号 *</text>
              <u-input v-model="addForm.version" placeholder="v1.0.0" border="bottom" />
            </view>
            <view class="form-item">
              <text class="form-label">描述</text>
              <u-textarea v-model="addForm.description" placeholder="算法描述" maxlength="500" />
            </view>
            <view class="form-item">
              <text class="form-label">模型路径</text>
              <u-input v-model="addForm.path" placeholder="模型文件路径" border="bottom" />
            </view>
            <view class="form-item">
              <text class="form-label">导入路径</text>
              <u-input v-model="addForm.importPath" placeholder="模型导入路径" border="bottom" />
            </view>
          </view>
          <view class="detail-footer">
            <u-button @click="addVisible = false">取消</u-button>
            <u-button type="primary" :loading="addSubmitting" @click="handleAdd">提交</u-button>
          </view>
        </scroll-view>
      </u-popup>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm, AlgorithmAuditForm } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";
import { getErrorMessage } from "@/utils/error";

// ==================== 工具定义 ====================

const STATUS_INFO: Record<number, { label: string; color: string }> = {
  1: { label: "草稿", color: "default" },
  2: { label: "测试中", color: "warning" },
  3: { label: "待审核", color: "primary" },
  4: { label: "已发布", color: "success" },
  5: { label: "已停用", color: "default" },
  6: { label: "已归档", color: "default" },
};

const statusFilters = [
  { label: "全部", value: "" },
  { label: "草稿", value: 1 },
  { label: "测试中", value: 2 },
  { label: "待审核", value: 3 },
  { label: "已发布", value: 4 },
  { label: "已停用", value: 5 },
];

const statusLabel = (s?: number) => STATUS_INFO[s ?? 0]?.label || "未知";

const statusTagType = (s?: number): "primary" | "success" | "warning" | "error" | "info" => {
  const map: Record<string, "primary" | "success" | "warning" | "error" | "info"> = {
    default: "info",
    primary: "primary",
    success: "success",
    warning: "warning",
    danger: "error",
  };
  return (map[STATUS_INFO[s ?? 0]?.color || "default"] as any) || "info";
};

const isDeletable = (s?: number) => s === 1 || s === 5 || s === 6;

// ==================== 树形工具 ====================

interface FlatNode {
  algorithm: Algorithm;
  level: number;
  hasChildren: boolean;
}

function flattenTree(nodes: Algorithm[], level = 0): FlatNode[] {
  const result: FlatNode[] = [];
  for (const node of nodes) {
    const hasChildren = !!(node.children && node.children.length > 0);
    result.push({ algorithm: node, level, hasChildren });
    if (hasChildren) result.push(...flattenTree(node.children!, level + 1));
  }
  return result;
}

function filterTree(nodes: Algorithm[], keyword: string, statusFilter: number | ""): Algorithm[] {
  const lowerKeyword = keyword.toLowerCase();
  const match = (algo: Algorithm): boolean => {
    const kwMatch = !keyword || (algo.name || "").toLowerCase().includes(lowerKeyword) || (algo.type || "").toLowerCase().includes(lowerKeyword);
    const sMatch = statusFilter === "" || algo.status === statusFilter;
    return kwMatch && sMatch;
  };
  const walk = (list: Algorithm[]): Algorithm[] => {
    const result: Algorithm[] = [];
    for (const node of list) {
      const children = node.children ? walk(node.children) : [];
      if (match(node) || children.length > 0) result.push({ ...node, children: children.length > 0 ? children : undefined });
    }
    return result;
  };
  return walk(nodes);
}

function updateAlgorithmInTree(nodes: Algorithm[], id: number, patch: Partial<Algorithm>): Algorithm[] {
  return nodes.map((node) => {
    if (node.id === id) return { ...node, ...patch };
    if (node.children) return { ...node, children: updateAlgorithmInTree(node.children, id, patch) };
    return node;
  });
}

function removeAlgorithmFromTree(nodes: Algorithm[], id: number): Algorithm[] {
  return nodes.filter((node) => node.id !== id).map((node) =>
    node.children ? { ...node, children: removeAlgorithmFromTree(node.children, id) } : node
  );
}

// ==================== 权限 ====================

const auth = useAuthStore();
const canAdd = computed(() => auth.hasPerm("sys:algorithm:add"));
const canAudit = computed(() => auth.hasPerm("sys:algorithm:audit"));
const canEdit = computed(() => auth.hasPerm("sys:algorithm:edit"));
const canDelete = computed(() => auth.hasPerm("sys:algorithm:delete"));

// ==================== 状态 ====================

const algorithms = ref<Algorithm[]>([]);
const loading = ref(true);
const keyword = ref("");
const statusFilter = ref<number | "">("");

const detailAlgo = ref<Algorithm | null>(null);
const detailVisible = ref(false);

const auditAlgo = ref<Algorithm | null>(null);
const auditVisible = ref(false);
const auditApproved = ref(true);
const auditRemark = ref("");
const auditSubmitting = ref(false);

const addVisible = ref(false);
const addForm = ref({ name: "", type: "", version: "", description: "", path: "", importPath: "" });
const addSubmitting = ref(false);

const actionLoadingId = ref<number | null>(null);

const flatList = computed(() => {
  const filtered = filterTree(algorithms.value, keyword.value, statusFilter.value);
  return flattenTree(filtered);
});

// ==================== 数据加载 ====================

async function fetchAlgorithms() {
  loading.value = true;
  try {
    const data = await AlgorithmAPI.getList();
    algorithms.value = data || [];
  } catch {
    algorithms.value = [];
  } finally {
    loading.value = false;
  }
}

fetchAlgorithms();

function handleSearch() {
  fetchAlgorithms();
}

// ==================== 详情 ====================

async function handleDetail(algo: Algorithm) {
  detailAlgo.value = algo;
  detailVisible.value = true;
  try {
    const detail = await AlgorithmAPI.getAlgorithmInfoById(algo.id);
    detailAlgo.value = detail;
  } catch {
    // fallback to list data
  }
}

// ==================== 停用/启用 ====================

async function handleToggleStatus(algo: Algorithm) {
  const isPublished = algo.status === 4;
  const newStatus = isPublished ? 5 : 4;
  const actionText = isPublished ? "停用" : "启用";
  const res = await uni.showModal({
    title: `确认${actionText}`,
    content: `确认${actionText}算法"${algo.name}"吗？`,
  });
  if (!res.confirm) return;
  actionLoadingId.value = algo.id;
  try {
    await AlgorithmAPI.updateStatus(algo.id, newStatus);
    algorithms.value = updateAlgorithmInTree(algorithms.value, algo.id, { status: newStatus });
    if (detailAlgo.value?.id === algo.id) detailAlgo.value = { ...detailAlgo.value, status: newStatus };
    uni.showToast({ title: `${actionText}成功`, icon: "success" });
  } catch (err: unknown) {
    uni.showToast({ title: getErrorMessage(err, `${actionText}失败`), icon: "error" });
  } finally {
    actionLoadingId.value = null;
  }
}

// ==================== 删除 ====================

async function handleDelete(algo: Algorithm) {
  const res = await uni.showModal({
    title: "确认删除",
    content: `确认删除算法"${algo.name}"吗？此操作不可恢复。`,
    confirmColor: "#ff4d4f",
  });
  if (!res.confirm) return;
  actionLoadingId.value = algo.id;
  try {
    await AlgorithmAPI.deleteByIds([String(algo.id)]);
    algorithms.value = removeAlgorithmFromTree(algorithms.value, algo.id);
    if (detailAlgo.value?.id === algo.id) detailAlgo.value = null;
    uni.showToast({ title: "删除成功", icon: "success" });
  } catch (err: unknown) {
    uni.showToast({ title: getErrorMessage(err, "删除失败"), icon: "error" });
  } finally {
    actionLoadingId.value = null;
  }
}

// ==================== 审核 ====================

function openAudit(algo: Algorithm, approved: boolean) {
  auditAlgo.value = algo;
  auditApproved.value = approved;
  auditRemark.value = "";
  auditVisible.value = true;
}

async function handleAuditSubmit() {
  if (!auditAlgo.value) return;
  if (!auditApproved.value && !auditRemark.value.trim()) {
    uni.showToast({ title: "驳回需填写原因", icon: "none" });
    return;
  }
  auditSubmitting.value = true;
  try {
    const form: AlgorithmAuditForm = {
      approved: auditApproved.value,
      remark: auditRemark.value.trim() || undefined,
    };
    await AlgorithmAPI.auditAlgorithm(auditAlgo.value.id, form);
    const newStatus = auditApproved.value ? 4 : 2;
    algorithms.value = updateAlgorithmInTree(algorithms.value, auditAlgo.value.id, { status: newStatus });
    if (detailAlgo.value?.id === auditAlgo.value.id) detailAlgo.value = { ...detailAlgo.value, status: newStatus };
    auditVisible.value = false;
    uni.showToast({ title: auditApproved.value ? "审核通过" : "已驳回", icon: "success" });
  } catch (err: unknown) {
    uni.showToast({ title: getErrorMessage(err, "审核失败"), icon: "error" });
  } finally {
    auditSubmitting.value = false;
  }
}

// ==================== 新增 ====================

function openAdd() {
  addForm.value = { name: "", type: "", version: "", description: "", path: "", importPath: "" };
  addVisible.value = true;
}

async function handleAdd() {
  const { name, type, version } = addForm.value;
  if (!name.trim() || !type.trim() || !version.trim()) {
    uni.showToast({ title: "名称/类型/版本为必填", icon: "none" });
    return;
  }
  if (!/^v?\d+\.\d+\.\d+$/.test(version.trim())) {
    uni.showToast({ title: "版本号格式: vX.Y.Z", icon: "none" });
    return;
  }
  addSubmitting.value = true;
  try {
    await AlgorithmAPI.add({
      name: name.trim(),
      type: type.trim(),
      version: version.trim(),
      description: addForm.value.description.trim() || undefined,
      path: addForm.value.path.trim() || undefined,
      importPath: addForm.value.importPath.trim() || undefined,
    } as Partial<Algorithm>);
    addVisible.value = false;
    uni.showToast({ title: "新增成功", icon: "success" });
    fetchAlgorithms();
  } catch (err: unknown) {
    uni.showToast({ title: getErrorMessage(err, "新增失败"), icon: "error" });
  } finally {
    addSubmitting.value = false;
  }
}
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}

// ==================== 搜索栏 ====================

.search-bar {
  display: flex;
  align-items: center;
  gap: 16rpx;
  margin-bottom: 20rpx;

  :deep(.u-search) {
    flex: 1;
  }

  .add-btn {
    flex-shrink: 0;
  }
}

// ==================== 筛选 ====================

.filter-row {
  display: flex;
  gap: 16rpx;
  margin-bottom: 20rpx;
  flex-wrap: wrap;
}

// ==================== 列表 ====================

.loading-wrapper {
  display: flex;
  justify-content: center;
  padding: 160rpx 0;
}

.empty-wrapper {
  padding: 160rpx 0;
}

.algo-list {
  display: flex;
  flex-direction: column;
  gap: 24rpx;
}

.algo-node {
  padding: 40rpx 48rpx;
  background: #fff;
  border-radius: 24rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.04);

  .node-main {
    display: flex;
    align-items: center;
    justify-content: space-between;

    .node-info {
      display: flex;
      flex: 1;
      gap: 16rpx;
      align-items: center;
      min-width: 0;

      .node-icon {
        flex-shrink: 0;
        font-size: 56rpx;
      }

      .node-name {
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 56rpx;
        font-weight: 500;
        color: #262626;
        white-space: nowrap;
      }
    }

    .node-meta {
      display: flex;
      flex-shrink: 0;
      gap: 16rpx;
      align-items: center;

      .node-type,
      .node-version {
        font-size: 44rpx;
        color: #8c8c8c;
      }
    }
  }

  .node-actions {
    display: flex;
    gap: 24rpx;
    justify-content: flex-end;
    padding-top: 32rpx;
    margin-top: 32rpx;
    border-top: 2rpx solid #f5f5f5;
  }
}

// ==================== 详情弹窗 ====================

.detail-content {
  max-height: 85vh;
  padding: 64rpx 48rpx;
}

.detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 48rpx;
  margin-bottom: 48rpx;
  border-bottom: 2rpx solid #f0f0f0;

  .detail-title {
    font-size: 68rpx;
    font-weight: 600;
    color: #262626;
  }

  .detail-close {
    font-size: 56rpx;
    color: #1890ff;
  }
}

.detail-section {
  margin-bottom: 64rpx;

  .section-title {
    display: block;
    padding-left: 24rpx;
    margin-bottom: 32rpx;
    font-size: 56rpx;
    font-weight: 600;
    color: #595959;
    border-left: 12rpx solid #1890ff;
  }
}

.detail-footer {
  display: flex;
  flex-direction: column;
  gap: 24rpx;
  padding-top: 64rpx;
}

// ==================== 审核弹窗 ====================

.audit-content {
  padding: 64rpx 48rpx;
  min-width: 500rpx;

  .audit-title {
    display: block;
    margin-bottom: 32rpx;
    font-size: 68rpx;
    font-weight: 600;
    color: #262626;
    text-align: center;
  }

  .audit-name {
    display: block;
    margin-bottom: 48rpx;
    font-size: 56rpx;
    color: #595959;
    text-align: center;
  }
}

.audit-remark {
  margin-bottom: 48rpx;

  .remark-label {
    display: block;
    margin-bottom: 24rpx;
    font-size: 52rpx;
    color: #595959;
  }
}

.audit-footer {
  display: flex;
  gap: 32rpx;
}

// ==================== 表单 ====================

.form-item {
  padding: 24rpx 0;

  .form-label {
    display: block;
    margin-bottom: 16rpx;
    font-size: 52rpx;
    color: #595959;
  }
}
</style>
