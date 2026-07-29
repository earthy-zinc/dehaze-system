<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="标题/内容/用户名"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="类型" prop="feedbackType">
          <el-select
            v-model="queryParams.feedbackType"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in typeOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-select
            v-model="queryParams.status"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in statusOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="优先级" prop="priority">
          <el-select
            v-model="queryParams.priority"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in priorityOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="处理人" prop="assigneeId">
          <el-input-number
            v-model="queryParams.assigneeId"
            :min="1"
            controls-position="right"
            placeholder="处理人ID"
            style="width: 140px"
          />
        </el-form-item>
        <el-form-item label="提交时间">
          <el-date-picker
            v-model="timeRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
            style="width: 240px"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><el-icon><Search /></el-icon>搜索</el-button
          >
          <el-button @click="resetQuery"
            ><el-icon><Refresh /></el-icon>重置</el-button
          >
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <el-button link type="primary" @click="goStats"
            ><el-icon><DataLine /></el-icon>反馈统计</el-button
          >
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />
        <el-table-column label="编号" prop="id" width="80" />
        <el-table-column
          label="标题"
          prop="title"
          min-width="200"
          show-overflow-tooltip
        />
        <el-table-column label="类型" width="100" align="center">
          <template #default="scope">
            <el-tag
              :type="typeTagType((scope.row as FeedbackPageVO).feedbackType)"
              size="small"
            >
              {{ typeLabel((scope.row as FeedbackPageVO).feedbackType) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="模块" width="120" align="center">
          <template #default="scope">
            <span>{{
              (scope.row as FeedbackPageVO).relatedModule || "-"
            }}</span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="scope">
            <el-tag
              :type="statusTagType((scope.row as FeedbackPageVO).status)"
              size="small"
            >
              {{ statusLabel((scope.row as FeedbackPageVO).status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="优先级" width="90" align="center">
          <template #default="scope">
            <el-tag
              :type="priorityTagType((scope.row as FeedbackPageVO).priority)"
              size="small"
            >
              {{ priorityLabel((scope.row as FeedbackPageVO).priority) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="处理人" width="100" align="center">
          <template #default="scope">
            <span v-if="(scope.row as FeedbackPageVO).assigneeName">
              {{ (scope.row as FeedbackPageVO).assigneeName }}
            </span>
            <span v-else style="color: #909399">未分配</span>
          </template>
        </el-table-column>
        <el-table-column label="提交时间" prop="createTime" width="170" />
        <el-table-column fixed="right" label="操作" width="320" align="center">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click="handleDetail(scope.row as FeedbackPageVO)"
            >
              <el-icon><View /></el-icon>详情
            </el-button>
            <el-button
              v-hasPerm="['feedback:assign']"
              link
              size="small"
              type="primary"
              :disabled="(scope.row as FeedbackPageVO).status === 'closed'"
              @click="openAssignDialog(scope.row as FeedbackPageVO)"
            >
              <el-icon><User /></el-icon>分配
            </el-button>
            <el-button
              v-hasPerm="['feedback:reply']"
              link
              size="small"
              type="primary"
              :disabled="(scope.row as FeedbackPageVO).status === 'closed'"
              @click="openReplyDialog(scope.row as FeedbackPageVO)"
            >
              <el-icon><ChatLineRound /></el-icon>回复
            </el-button>
            <el-button
              v-hasPerm="['feedback:edit']"
              link
              size="small"
              type="primary"
              @click="openTagDialog(scope.row as FeedbackPageVO)"
            >
              <el-icon><CollectionTag /></el-icon>标签
            </el-button>
            <el-button
              v-if="(scope.row as FeedbackPageVO).status !== 'closed'"
              v-hasPerm="['feedback:close']"
              link
              size="small"
              type="danger"
              @click="handleClose(scope.row as FeedbackPageVO)"
            >
              <el-icon><CircleClose /></el-icon>关闭
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="total > 0"
        v-model:limit="queryParams.pageSize"
        v-model:page="queryParams.pageNum"
        v-model:total="total"
        @pagination="handleQuery"
      />
    </el-card>

    <!-- 详情弹窗 -->
    <el-dialog
      v-model="detailDialog.visible"
      title="反馈详情"
      width="780px"
      @close="detailDialog.data = null"
    >
      <div v-loading="detailDialog.loading">
        <el-tabs v-if="detailDialog.data">
          <el-tab-pane label="基本信息">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="标题">{{
                detailDialog.data.title
              }}</el-descriptions-item>
              <el-descriptions-item label="类型">
                <el-tag
                  :type="typeTagType(detailDialog.data.feedbackType)"
                  size="small"
                >
                  {{ typeLabel(detailDialog.data.feedbackType) }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="状态">
                <el-tag
                  :type="statusTagType(detailDialog.data.status)"
                  size="small"
                >
                  {{ statusLabel(detailDialog.data.status) }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="优先级">
                <el-tag
                  :type="priorityTagType(detailDialog.data.priority)"
                  size="small"
                >
                  {{ priorityLabel(detailDialog.data.priority) }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="模块">{{
                detailDialog.data.relatedModule || "-"
              }}</el-descriptions-item>
              <el-descriptions-item label="处理人">{{
                detailDialog.data.assigneeName || "未分配"
              }}</el-descriptions-item>
              <el-descriptions-item label="提交时间">{{
                detailDialog.data.createTime
              }}</el-descriptions-item>
              <el-descriptions-item label="联系方式">
                <el-tag
                  v-if="detailDialog.data.contact"
                  type="info"
                  size="small"
                  >{{ detailDialog.data.contact }}</el-tag
                >
                <span v-else>-</span>
              </el-descriptions-item>
              <el-descriptions-item label="标签" :span="2">
                <el-tag
                  v-for="tag in detailDialog.data.tags || []"
                  :key="tag"
                  size="small"
                  style="margin-right: 6px"
                  >{{ tag }}</el-tag
                >
                <span
                  v-if="
                    !detailDialog.data.tags ||
                    detailDialog.data.tags.length === 0
                  "
                  >-</span
                >
              </el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>
          <el-tab-pane label="内容与回复">
            <el-card shadow="never" style="margin-bottom: 16px">
              <template #header>反馈内容</template>
              <div style="white-space: pre-wrap">
                {{ detailDialog.data.content }}
              </div>
              <div
                v-if="
                  detailDialog.data.images &&
                  detailDialog.data.images.length > 0
                "
                style="margin-top: 12px"
              >
                <el-image
                  v-for="(img, idx) in detailDialog.data.images"
                  :key="idx"
                  :src="img"
                  :preview-src-list="detailDialog.data.images"
                  :initial-index="idx"
                  fit="cover"
                  style="width: 100px; height: 100px; margin-right: 8px"
                />
              </div>
            </el-card>

            <el-timeline
              v-if="
                detailDialog.data.replies &&
                detailDialog.data.replies.length > 0
              "
            >
              <el-timeline-item
                v-for="reply in detailDialog.data.replies"
                :key="reply.id"
                :timestamp="reply.createTime"
                placement="top"
                :color="reply.replierType === 2 ? '#409eff' : '#67c23a'"
              >
                <div style="margin-bottom: 6px">
                  <span style="font-weight: 600">{{ reply.replierName }}</span>
                  <el-tag
                    v-if="reply.replyType"
                    :type="replyTypeTagType(reply.replyType)"
                    size="small"
                    style="margin-left: 8px"
                    >{{ replyTypeLabel(reply.replyType) }}</el-tag
                  >
                </div>
                <div style="white-space: pre-wrap">{{ reply.content }}</div>
                <div
                  v-if="reply.attachments && reply.attachments.length > 0"
                  style="margin-top: 8px"
                >
                  <el-image
                    v-for="(att, idx) in reply.attachments"
                    :key="idx"
                    :src="att"
                    :preview-src-list="reply.attachments"
                    :initial-index="idx"
                    fit="cover"
                    style="width: 80px; height: 80px; margin-right: 8px"
                  />
                </div>
              </el-timeline-item>
            </el-timeline>

            <el-card
              v-if="detailDialog.data.closeReason"
              shadow="never"
              style="margin-top: 16px"
            >
              <template #header>关闭原因</template>
              <div style="white-space: pre-wrap">
                {{ detailDialog.data.closeReason }}
              </div>
            </el-card>
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-dialog>

    <!-- 分配弹窗 -->
    <el-dialog
      v-model="assignDialog.visible"
      title="分配处理人"
      width="460px"
      @close="resetAssignForm"
    >
      <el-form
        ref="assignFormRef"
        :model="assignForm"
        :rules="assignRules"
        label-width="100px"
      >
        <el-form-item label="处理人ID" prop="assigneeId">
          <el-input-number
            v-model="assignForm.assigneeId"
            :min="1"
            controls-position="right"
            style="width: 200px"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleAssignSubmit"
            >确 定</el-button
          >
          <el-button @click="assignDialog.visible = false">取 消</el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 回复弹窗 -->
    <el-dialog
      v-model="replyDialog.visible"
      title="回复反馈"
      width="560px"
      @close="resetReplyForm"
    >
      <el-form
        ref="replyFormRef"
        :model="replyForm"
        :rules="replyRules"
        label-width="100px"
      >
        <el-form-item label="回复类型" prop="replyType">
          <el-select
            v-model="replyForm.replyType"
            placeholder="请选择"
            style="width: 200px"
          >
            <el-option
              v-for="opt in replyTypeOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="回复内容" prop="content">
          <el-input
            v-model="replyForm.content"
            type="textarea"
            :rows="4"
            placeholder="请输入回复内容（10-2000 字符）"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleReplySubmit">确 定</el-button>
          <el-button @click="replyDialog.visible = false">取 消</el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 标签编辑弹窗 -->
    <el-dialog v-model="tagDialog.visible" title="编辑标签" width="500px">
      <el-form label-width="80px">
        <el-form-item label="标签">
          <el-select
            v-model="tagDialog.tags"
            multiple
            filterable
            allow-create
            default-first-option
            placeholder="输入标签后回车"
            style="width: 100%"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleTagSubmit">确 定</el-button>
          <el-button @click="tagDialog.visible = false">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  FeedbackAPI,
  FeedbackQuery,
  FeedbackPageVO,
  FeedbackDetailVO,
  FeedbackAssignForm,
  FeedbackReplyForm,
  FeedbackCloseForm,
  FeedbackStatus,
  FeedbackType,
  FeedbackReplyType,
} from "dehaze-sdk-js";
import {
  Search,
  Refresh,
  View,
  User,
  ChatLineRound,
  CircleClose,
  CollectionTag,
  DataLine,
} from "@element-plus/icons-vue";

defineOptions({
  name: "FeedbackList",
  inheritAttrs: false,
});

const router = useRouter();

const queryFormRef = ref(ElForm);
const assignFormRef = ref(ElForm);
const replyFormRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);
const ids = ref<number[]>([]);
const timeRange = ref<[string, string] | null>(null);

const queryParams = reactive<FeedbackQuery>({
  pageNum: 1,
  pageSize: 10,
});

const pageData = ref<FeedbackPageVO[]>([]);

const typeOptions: { label: string; value: FeedbackType }[] = [
  { label: "功能建议", value: "suggestion" },
  { label: "问题报告", value: "bug" },
  { label: "体验反馈", value: "experience" },
  { label: "投诉", value: "complaint" },
];

const statusOptions: { label: string; value: FeedbackStatus }[] = [
  { label: "待处理", value: "pending" },
  { label: "处理中", value: "processing" },
  { label: "已回复", value: "replied" },
  { label: "已关闭", value: "closed" },
];

const priorityOptions: { label: string; value: number }[] = [
  { label: "低", value: 1 },
  { label: "中", value: 2 },
  { label: "高", value: 3 },
  { label: "紧急", value: 4 },
];

const replyTypeOptions: { label: string; value: FeedbackReplyType }[] = [
  { label: "通知", value: "info" },
  { label: "已解决", value: "resolved" },
  { label: "不支持", value: "unsupported" },
  { label: "转开发", value: "dev_transfer" },
];

function typeLabel(type: FeedbackType): string {
  const map: Record<FeedbackType, string> = {
    suggestion: "功能建议",
    bug: "问题报告",
    experience: "体验反馈",
    complaint: "投诉",
  };
  return map[type] || type;
}

function typeTagType(
  type: FeedbackType
): "primary" | "danger" | "success" | "warning" {
  const map: Record<
    FeedbackType,
    "primary" | "danger" | "success" | "warning"
  > = {
    suggestion: "primary",
    bug: "danger",
    experience: "success",
    complaint: "warning",
  };
  return map[type];
}

function statusLabel(status: FeedbackStatus): string {
  const map: Record<FeedbackStatus, string> = {
    pending: "待处理",
    processing: "处理中",
    replied: "已回复",
    closed: "已关闭",
  };
  return map[status] || status;
}

function statusTagType(
  status: FeedbackStatus
): "warning" | "primary" | "success" | "info" {
  const map: Record<
    FeedbackStatus,
    "warning" | "primary" | "success" | "info"
  > = {
    pending: "warning",
    processing: "primary",
    replied: "success",
    closed: "info",
  };
  return map[status];
}

function priorityLabel(priority: number): string {
  const map: Record<number, string> = { 1: "低", 2: "中", 3: "高", 4: "紧急" };
  return map[priority] || String(priority);
}

function priorityTagType(
  priority: number
): "info" | "primary" | "warning" | "danger" {
  const map: Record<number, "info" | "primary" | "warning" | "danger"> = {
    1: "info",
    2: "primary",
    3: "warning",
    4: "danger",
  };
  return map[priority] || "info";
}

function replyTypeLabel(type: FeedbackReplyType): string {
  const map: Record<FeedbackReplyType, string> = {
    info: "通知",
    resolved: "已解决",
    unsupported: "不支持",
    dev_transfer: "转开发",
  };
  return map[type] || type;
}

function replyTypeTagType(
  type: FeedbackReplyType
): "info" | "success" | "warning" {
  const map: Record<FeedbackReplyType, "info" | "success" | "warning"> = {
    info: "info",
    resolved: "success",
    unsupported: "info",
    dev_transfer: "warning",
  };
  return map[type];
}

function handleQuery() {
  loading.value = true;
  if (timeRange.value && timeRange.value.length === 2) {
    queryParams.startTime = timeRange.value[0];
    queryParams.endTime = timeRange.value[1];
  } else {
    queryParams.startTime = undefined;
    queryParams.endTime = undefined;
  }
  FeedbackAPI.listFeedback(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value?.resetFields();
  timeRange.value = null;
  queryParams.keywords = undefined;
  queryParams.feedbackType = undefined;
  queryParams.status = undefined;
  queryParams.priority = undefined;
  queryParams.assigneeId = undefined;
  queryParams.startTime = undefined;
  queryParams.endTime = undefined;
  queryParams.pageNum = 1;
  handleQuery();
}

function handleSelectionChange(selection: any) {
  ids.value = selection.map((item: any) => item.id);
}

function goStats() {
  router.push("/feedback/stats?tab=feedback");
}

// 详情弹窗
const detailDialog = reactive<{
  visible: boolean;
  loading: boolean;
  data: FeedbackDetailVO | null;
}>({
  visible: false,
  loading: false,
  data: null,
});

function handleDetail(row: FeedbackPageVO) {
  detailDialog.visible = true;
  detailDialog.loading = true;
  detailDialog.data = null;
  FeedbackAPI.getFeedbackDetail(row.id)
    .then((data) => {
      detailDialog.data = data;
    })
    .finally(() => {
      detailDialog.loading = false;
    });
}

// 分配弹窗
const assignDialog = reactive<{
  visible: boolean;
  loading: boolean;
  id: number;
}>({
  visible: false,
  loading: false,
  id: 0,
});

const assignForm = reactive<FeedbackAssignForm>({
  assigneeId: 1,
});

const assignRules = reactive({
  assigneeId: [{ required: true, message: "请输入处理人ID", trigger: "blur" }],
});

function openAssignDialog(row: FeedbackPageVO) {
  assignDialog.id = row.id;
  assignForm.assigneeId = row.assigneeId ?? 1;
  assignDialog.visible = true;
}

function resetAssignForm() {
  assignFormRef.value?.resetFields();
  assignForm.assigneeId = 1;
}

function handleAssignSubmit() {
  assignFormRef.value.validate((valid: any) => {
    if (!valid) return;
    assignDialog.loading = true;
    FeedbackAPI.assignFeedback(assignDialog.id, {
      assigneeId: assignForm.assigneeId,
    })
      .then(() => {
        ElMessage.success("分配成功");
        assignDialog.visible = false;
        handleQuery();
      })
      .finally(() => {
        assignDialog.loading = false;
      });
  });
}

// 回复弹窗
const replyDialog = reactive<{
  visible: boolean;
  loading: boolean;
  id: number;
}>({
  visible: false,
  loading: false,
  id: 0,
});

const replyForm = reactive<FeedbackReplyForm>({
  replyType: "info",
  content: "",
});

const replyRules = reactive({
  replyType: [{ required: true, message: "请选择回复类型", trigger: "change" }],
  content: [
    { required: true, message: "请输入回复内容", trigger: "blur" },
    {
      min: 10,
      max: 2000,
      message: "回复内容长度为 10-2000 字符",
      trigger: "blur",
    },
  ],
});

function openReplyDialog(row: FeedbackPageVO) {
  replyDialog.id = row.id;
  replyForm.replyType = "info";
  replyForm.content = "";
  replyDialog.visible = true;
}

function resetReplyForm() {
  replyFormRef.value?.resetFields();
  replyForm.replyType = "info";
  replyForm.content = "";
}

function handleReplySubmit() {
  replyFormRef.value.validate((valid: any) => {
    if (!valid) return;
    replyDialog.loading = true;
    FeedbackAPI.replyFeedback(replyDialog.id, {
      replyType: replyForm.replyType,
      content: replyForm.content,
    })
      .then(() => {
        ElMessage.success("回复成功");
        replyDialog.visible = false;
        handleQuery();
      })
      .finally(() => {
        replyDialog.loading = false;
      });
  });
}

// 关闭反馈
function handleClose(row: FeedbackPageVO) {
  ElMessageBox.prompt("请输入关闭反馈的原因", "关闭反馈", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    inputType: "textarea",
    inputPlaceholder: "请输入关闭原因",
    inputValidator: (val: string) => {
      if (!val || !val.trim()) return "关闭原因不能为空";
      return true;
    },
    lockScroll: false,
  })
    .then(({ value }) => {
      const form: FeedbackCloseForm = { closeReason: value.trim() };
      return FeedbackAPI.closeFeedback(row.id, form);
    })
    .then(() => {
      ElMessage.success("已关闭");
      handleQuery();
    })
    .catch(() => {});
}

// 标签弹窗
const tagDialog = reactive<{
  visible: boolean;
  loading: boolean;
  id: number;
  tags: string[];
}>({
  visible: false,
  loading: false,
  id: 0,
  tags: [],
});

function openTagDialog(row: FeedbackPageVO) {
  tagDialog.id = row.id;
  tagDialog.tags = row.tags ? [...row.tags] : [];
  tagDialog.visible = true;
}

function handleTagSubmit() {
  tagDialog.loading = true;
  FeedbackAPI.updateFeedbackTags(tagDialog.id, tagDialog.tags)
    .then(() => {
      ElMessage.success("标签更新成功");
      if (detailDialog.data && detailDialog.data.id === tagDialog.id) {
        detailDialog.data = { ...detailDialog.data, tags: [...tagDialog.tags] };
      }
      tagDialog.visible = false;
      handleQuery();
    })
    .finally(() => {
      tagDialog.loading = false;
    });
}

onMounted(() => {
  handleQuery();
});
</script>
