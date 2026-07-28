<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="公告标题" prop="title">
          <el-input
            v-model="queryParams.title"
            clearable
            placeholder="公告标题"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="公告类型" prop="type">
          <el-select
            v-model="queryParams.type"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="item in typeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
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
              v-for="item in statusOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <el-icon><Search /></el-icon>搜索
          </el-button>
          <el-button @click="resetQuery">
            <el-icon><Refresh /></el-icon>重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <div>
            <el-button
              v-hasPerm="['notify:announcement:add']"
              type="success"
              @click="openDialog()"
            >
              <el-icon><Plus /></el-icon>新增公告
            </el-button>
          </div>
        </div>
      </template>

      <el-table
        ref="dataTableRef"
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
      >
        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column
          label="公告标题"
          prop="title"
          min-width="200"
          show-overflow-tooltip
        />
        <el-table-column label="类型" width="110" align="center">
          <template #default="scope">
            <span :class="['type-tag', `tag-${scope.row.type}`]">
              {{ scope.row.typeLabel }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="重要级别" width="100" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.importance === 2 ? 'danger' : 'info'"
              size="small"
              effect="plain"
            >
              {{ scope.row.importance === 2 ? "重要" : "普通" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="发送范围"
          prop="targetScopeLabel"
          width="120"
          align="center"
        />
        <el-table-column label="状态" width="100" align="center">
          <template #default="scope">
            <span :class="['status-tag', `status-${scope.row.status}`]">
              {{ scope.row.statusLabel }}
            </span>
          </template>
        </el-table-column>
        <el-table-column
          label="发送时间"
          prop="sendTime"
          width="170"
          align="center"
        />
        <el-table-column
          label="送达数"
          prop="sentCount"
          width="90"
          align="center"
        >
          <template #default="scope">
            {{ scope.row.sentCount ?? "-" }}
          </template>
        </el-table-column>
        <el-table-column
          label="创建时间"
          prop="createTime"
          width="170"
          align="center"
        />
        <el-table-column fixed="right" label="操作" width="240" align="center">
          <template #default="scope">
            <el-button
              v-if="scope.row.status === 1 || scope.row.status === 2"
              v-hasPerm="['notify:announcement:edit']"
              link
              size="small"
              type="primary"
              @click="openDialog(scope.row.id)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-if="scope.row.status === 1 || scope.row.status === 2"
              v-hasPerm="['notify:announcement:send']"
              link
              size="small"
              type="success"
              @click="handleSend(scope.row as AnnouncementVO)"
            >
              <el-icon><Promotion /></el-icon>发送
            </el-button>
            <el-button
              v-if="scope.row.status === 2"
              v-hasPerm="['notify:announcement:cancel']"
              link
              size="small"
              type="warning"
              @click="handleCancel(scope.row as AnnouncementVO)"
            >
              <el-icon><CircleClose /></el-icon>取消
            </el-button>
            <el-button
              v-hasPerm="['notify:announcement:delete']"
              link
              size="small"
              type="danger"
              @click="handleDelete(scope.row as AnnouncementVO)"
            >
              <el-icon><Delete /></el-icon>删除
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

    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="640px"
      append-to-body
      @close="closeDialog"
    >
      <el-form
        ref="formRef"
        :model="formData"
        :rules="rules"
        label-width="100px"
      >
        <el-form-item label="公告标题" prop="title">
          <el-input
            v-model="formData.title"
            placeholder="2-50 字符"
            maxlength="50"
            show-word-limit
          />
        </el-form-item>
        <el-form-item label="公告内容" prop="content">
          <el-input
            v-model="formData.content"
            type="textarea"
            :rows="5"
            placeholder="请输入公告内容"
          />
        </el-form-item>
        <el-form-item label="公告类型" prop="type">
          <el-select
            v-model="formData.type"
            placeholder="请选择"
            style="width: 100%"
          >
            <el-option
              v-for="item in typeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="重要级别" prop="importance">
          <el-radio-group v-model="formData.importance">
            <el-radio :label="1">普通</el-radio>
            <el-radio :label="2">重要</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="发送范围" prop="targetScope">
          <el-select v-model="formData.targetScope" style="width: 100%">
            <el-option
              v-for="item in targetScopeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item
          v-if="formData.targetScope === 'level'"
          label="会员等级"
          prop="targetLevel"
        >
          <el-input-number
            v-model="targetLevel"
            :min="1"
            :max="10"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item
          v-if="formData.targetScope === 'specified'"
          label="用户ID"
          prop="targetUserIds"
        >
          <el-input
            v-model="targetUserIdsStr"
            placeholder="多个用英文逗号分隔，如 1,2,3"
          />
        </el-form-item>
        <el-form-item label="定时发送" prop="sendTime">
          <el-date-picker
            v-model="formData.sendTime"
            type="datetime"
            format="YYYY-MM-DD HH:mm:ss"
            value-format="YYYY-MM-DD HH:mm:ss"
            placeholder="留空则保存为草稿"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="过期时间" prop="expireTime">
          <el-date-picker
            v-model="formData.expireTime"
            type="datetime"
            format="YYYY-MM-DD HH:mm:ss"
            value-format="YYYY-MM-DD HH:mm:ss"
            placeholder="可选"
            style="width: 100%"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  AnnouncementAPI,
  AnnouncementForm,
  AnnouncementQuery,
  AnnouncementVO,
} from "dehaze-sdk-js";
import {
  CircleClose,
  Delete,
  Edit,
  Plus,
  Promotion,
  Refresh,
  Search,
} from "@element-plus/icons-vue";

defineOptions({ name: "NotifyAnnouncement" });

const queryFormRef = ref(ElForm);
const formRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);
const pageData = ref<AnnouncementVO[]>([]);
const queryParams = reactive<AnnouncementQuery>({
  pageNum: 1,
  pageSize: 10,
});

const typeOptions = [
  { value: "maintenance", label: "系统维护" },
  { value: "feature", label: "功能更新" },
  { value: "activity", label: "活动通知" },
  { value: "operation", label: "运营公告" },
];

const statusOptions = [
  { value: 1, label: "草稿" },
  { value: 2, label: "待发送" },
  { value: 3, label: "已发送" },
  { value: 4, label: "已取消" },
];

const targetScopeOptions = [
  { value: "all", label: "全体用户" },
  { value: "level", label: "按会员等级" },
  { value: "specified", label: "指定用户" },
];

const dialog = reactive({
  visible: false,
  title: "",
  isEdit: false,
  editId: 0,
});

const defaultForm = (): AnnouncementForm => ({
  title: "",
  content: "",
  type: "maintenance",
  importance: 1,
  targetScope: "all",
});

const formData = reactive<AnnouncementForm>(defaultForm());
const targetLevel = ref(1);
const targetUserIdsStr = ref("");

const rules = reactive({
  title: [
    { required: true, message: "请输入公告标题", trigger: "blur" },
    { min: 2, max: 50, message: "标题长度 2-50 字符", trigger: "blur" },
  ],
  content: [{ required: true, message: "请输入公告内容", trigger: "blur" }],
  type: [{ required: true, message: "请选择公告类型", trigger: "change" }],
  importance: [
    { required: true, message: "请选择重要级别", trigger: "change" },
  ],
  targetScope: [
    { required: true, message: "请选择发送范围", trigger: "change" },
  ],
});

function handleQuery() {
  loading.value = true;
  AnnouncementAPI.getPage(queryParams)
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
  queryParams.pageNum = 1;
  handleQuery();
}

function openDialog(id?: number) {
  resetForm();
  if (id) {
    dialog.title = "编辑公告";
    dialog.isEdit = true;
    dialog.editId = id;
    AnnouncementAPI.getDetail(id).then((data) => {
      Object.assign(formData, {
        title: data.title,
        content: data.content ?? "",
        type: data.type,
        importance: data.importance,
        targetScope: data.targetScope,
        sendTime: data.sendTime,
        expireTime: data.expireTime,
      });
      if (data.targetParams) {
        if (data.targetScope === "level") {
          targetLevel.value = data.targetParams.level ?? 1;
        } else if (data.targetScope === "specified") {
          targetUserIdsStr.value = (data.targetParams.userIds ?? []).join(",");
        }
      }
    });
  } else {
    dialog.title = "新增公告";
    dialog.isEdit = false;
    dialog.editId = 0;
  }
  dialog.visible = true;
}

function resetForm() {
  Object.assign(formData, defaultForm());
  targetLevel.value = 1;
  targetUserIdsStr.value = "";
  formRef.value?.clearValidate();
}

function closeDialog() {
  dialog.visible = false;
  resetForm();
}

function buildTargetParams(): Record<string, any> | undefined {
  if (formData.targetScope === "level") {
    return { level: targetLevel.value };
  }
  if (formData.targetScope === "specified") {
    const ids = targetUserIdsStr.value
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .map(Number)
      .filter((n) => !isNaN(n) && n > 0);
    return { userIds: ids };
  }
  return undefined;
}

function handleSubmit() {
  formRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    const payload: AnnouncementForm = {
      ...formData,
      targetParams: buildTargetParams(),
    };
    if (dialog.isEdit) {
      AnnouncementAPI.update(dialog.editId, payload).then(() => {
        ElMessage.success("修改成功");
        closeDialog();
        handleQuery();
      });
    } else {
      AnnouncementAPI.create(payload).then(() => {
        ElMessage.success("新增成功");
        closeDialog();
        handleQuery();
      });
    }
  });
}

function handleDelete(row: AnnouncementVO) {
  ElMessageBox.confirm(`确定删除公告「${row.title}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  })
    .then(() => AnnouncementAPI.deleteById(row.id))
    .then(() => {
      ElMessage.success("删除成功");
      handleQuery();
    })
    .catch(() => {});
}

function handleSend(row: AnnouncementVO) {
  ElMessageBox.confirm(`确定立即发送公告「${row.title}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  })
    .then(() => AnnouncementAPI.send(row.id))
    .then((res) => {
      ElMessage.success(`发送成功，共送达 ${res.sentCount} 位用户`);
      handleQuery();
    })
    .catch(() => {});
}

function handleCancel(row: AnnouncementVO) {
  ElMessageBox.confirm(`确定取消定时公告「${row.title}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  })
    .then(() => AnnouncementAPI.cancel(row.id))
    .then(() => {
      ElMessage.success("取消成功");
      handleQuery();
    })
    .catch(() => {});
}

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.type-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  border-radius: 4px;

  &.tag-maintenance {
    color: #722ed1;
    background: #f9f0ff;
  }

  &.tag-feature {
    color: #52c41a;
    background: #f6ffed;
  }

  &.tag-activity {
    color: #eb2f96;
    background: #fff0f6;
  }

  &.tag-operation {
    color: #13c2c2;
    background: #e6fffb;
  }
}

.status-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  border-radius: 10px;

  &.status-1 {
    color: #8c8c8c;
    background: #fafafa;
  }

  &.status-2 {
    color: #fa8c16;
    background: #fff7e6;
  }

  &.status-3 {
    color: #52c41a;
    background: #f6ffed;
  }

  &.status-4 {
    color: #8c8c8c;
    background: #fafafa;
  }
}
</style>
