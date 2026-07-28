<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="模板名称" prop="name">
          <el-input
            v-model="queryParams.name"
            clearable
            placeholder="模板名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="消息类型" prop="type">
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
            style="width: 120px"
          >
            <el-option :value="1" label="启用" />
            <el-option :value="0" label="禁用" />
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
      <el-table
        ref="dataTableRef"
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
      >
        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column label="模板编码" prop="code" width="180" />
        <el-table-column
          label="模板名称"
          prop="name"
          min-width="160"
          show-overflow-tooltip
        />
        <el-table-column label="类型" width="110" align="center">
          <template #default="scope">
            <span :class="['type-tag', `tag-${scope.row.type}`]">
              {{ getTypeLabel(scope.row.type) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column
          label="标题模板"
          prop="titleTemplate"
          min-width="200"
          show-overflow-tooltip
        />
        <el-table-column label="优先级" width="90" align="center">
          <template #default="scope">
            <el-tag
              :type="priorityTagType(scope.row.priority)"
              size="small"
              effect="plain"
            >
              {{ priorityLabel(scope.row.priority) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="90" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.status === 1 ? 'success' : 'info'"
              size="small"
            >
              {{ scope.row.status === 1 ? "启用" : "禁用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="更新时间"
          prop="updateTime"
          width="170"
          align="center"
        >
          <template #default="scope">
            {{ scope.row.updateTime || scope.row.createTime }}
          </template>
        </el-table-column>
        <el-table-column fixed="right" label="操作" width="160" align="center">
          <template #default="scope">
            <el-button
              v-hasPerm="['notify:template:edit']"
              link
              size="small"
              type="primary"
              @click="openDialog(scope.row.id)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click="openDetail(scope.row.id)"
            >
              <el-icon><View /></el-icon>详情
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
      width="680px"
      append-to-body
      @close="closeDialog"
    >
      <el-form ref="formRef" :model="formData" label-width="100px">
        <el-form-item label="模板名称">
          <el-input v-model="formData.name" placeholder="请输入模板名称" />
        </el-form-item>
        <el-form-item label="标题模板">
          <el-input
            v-model="formData.titleTemplate"
            type="textarea"
            :rows="2"
            placeholder="支持变量占位，如：恭喜您升级至 {levelName}"
          />
        </el-form-item>
        <el-form-item label="正文模板">
          <el-input
            v-model="formData.contentTemplate"
            type="textarea"
            :rows="6"
            placeholder="支持变量占位，如：您已解锁以下新权益：{benefitList}"
          />
        </el-form-item>
        <el-form-item label="默认优先级">
          <el-radio-group v-model="formData.priority">
            <el-radio :label="1">低</el-radio>
            <el-radio :label="2">中</el-radio>
            <el-radio :label="3">高</el-radio>
            <el-radio :label="4">紧急</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="推送渠道">
          <el-checkbox-group v-model="channels">
            <el-checkbox label="inbox">站内信</el-checkbox>
            <el-checkbox label="push">APP 推送</el-checkbox>
            <el-checkbox label="email">邮件</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item label="状态">
          <el-radio-group v-model="formData.status">
            <el-radio :label="1">启用</el-radio>
            <el-radio :label="0">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>

    <el-dialog
      v-model="detailDialog.visible"
      title="模板详情"
      width="680px"
      append-to-body
    >
      <template v-if="detailData">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="模板编码">
            {{ detailData.code }}
          </el-descriptions-item>
          <el-descriptions-item label="模板名称">
            {{ detailData.name }}
          </el-descriptions-item>
          <el-descriptions-item label="消息类型">
            {{ getTypeLabel(detailData.type) }}
          </el-descriptions-item>
          <el-descriptions-item label="标题模板">
            <code class="template-code">{{ detailData.titleTemplate }}</code>
          </el-descriptions-item>
          <el-descriptions-item
            v-if="detailData.contentTemplate"
            label="正文模板"
          >
            <pre class="template-pre">{{ detailData.contentTemplate }}</pre>
          </el-descriptions-item>
          <el-descriptions-item label="优先级">
            {{ priorityLabel(detailData.priority) }}
          </el-descriptions-item>
          <el-descriptions-item v-if="detailData.channels" label="推送渠道">
            {{ formatChannels(detailData.channels) }}
          </el-descriptions-item>
          <el-descriptions-item
            v-if="detailData.variables?.length"
            label="模板变量"
          >
            <div class="variable-list">
              <span
                v-for="v in detailData.variables"
                :key="v.name"
                class="variable-item"
              >
                <code>{{ "{" + v.name + "}" }}</code>
                <span class="variable-desc">{{ v.desc }}</span>
              </span>
            </div>
          </el-descriptions-item>
        </el-descriptions>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  MessageTemplateAPI,
  MessageTemplateForm,
  MessageTemplateQuery,
  MessageTemplateVO,
} from "dehaze-sdk-js";
import { Edit, Refresh, Search, View } from "@element-plus/icons-vue";

defineOptions({ name: "NotifyTemplate" });

const queryFormRef = ref(ElForm);
const formRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);
const pageData = ref<MessageTemplateVO[]>([]);
const queryParams = reactive<MessageTemplateQuery>({
  pageNum: 1,
  pageSize: 20,
});

const typeOptions = [
  { value: "inbox", label: "站内信" },
  { value: "announcement", label: "系统公告" },
  { value: "business", label: "业务通知" },
  { value: "member", label: "会员通知" },
  { value: "alert", label: "告警通知" },
  { value: "critical_alert", label: "严重告警" },
];

function getTypeLabel(type: string) {
  return typeOptions.find((t) => t.value === type)?.label ?? type;
}

function priorityLabel(p: number) {
  const map: Record<number, string> = { 1: "低", 2: "中", 3: "高", 4: "紧急" };
  return map[p] ?? String(p);
}

function priorityTagType(
  p: number
): "info" | "warning" | "primary" | "success" | "danger" {
  const map: Record<
    number,
    "info" | "warning" | "primary" | "success" | "danger"
  > = {
    1: "info",
    2: "primary",
    3: "warning",
    4: "danger",
  };
  return map[p] ?? "info";
}

function formatChannels(channels: Record<string, boolean>) {
  const map: Record<string, string> = {
    inbox: "站内信",
    push: "APP 推送",
    email: "邮件",
  };
  return Object.entries(channels)
    .filter(([, v]) => v)
    .map(([k]) => map[k] ?? k)
    .join("、");
}

const dialog = reactive({
  visible: false,
  title: "",
  editId: 0,
});

const formData = reactive<MessageTemplateForm>({
  name: "",
  titleTemplate: "",
  contentTemplate: "",
  priority: 2,
  status: 1,
});

const channels = ref<string[]>([]);

const detailDialog = reactive({ visible: false });
const detailData = ref<MessageTemplateVO | null>(null);

function handleQuery() {
  loading.value = true;
  MessageTemplateAPI.getPage(queryParams)
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

function openDialog(id: number) {
  dialog.editId = id;
  dialog.title = "编辑模板";
  MessageTemplateAPI.getDetail(id).then((data) => {
    formData.name = data.name;
    formData.titleTemplate = data.titleTemplate;
    formData.contentTemplate = data.contentTemplate ?? "";
    formData.priority = data.priority;
    formData.status = data.status;
    channels.value = data.channels
      ? Object.entries(data.channels)
          .filter(([, v]) => v)
          .map(([k]) => k)
      : [];
    dialog.visible = true;
  });
}

function closeDialog() {
  dialog.visible = false;
  Object.assign(formData, {
    name: "",
    titleTemplate: "",
    contentTemplate: "",
    priority: 2,
    status: 1,
  });
  channels.value = [];
  formRef.value?.clearValidate();
}

function openDetail(id: number) {
  MessageTemplateAPI.getDetail(id).then((data) => {
    detailData.value = data;
    detailDialog.visible = true;
  });
}

function handleSubmit() {
  const channelsMap: Record<string, boolean> = {
    inbox: channels.value.includes("inbox"),
    push: channels.value.includes("push"),
    email: channels.value.includes("email"),
  };
  MessageTemplateAPI.update(dialog.editId, {
    ...formData,
    channels: channelsMap,
  }).then(() => {
    ElMessage.success("保存成功");
    closeDialog();
    handleQuery();
  });
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

  &.tag-inbox {
    color: #8c8c8c;
    background: #fafafa;
  }

  &.tag-announcement {
    color: #409eff;
    background: #ecf5ff;
  }

  &.tag-business {
    color: #13c2c2;
    background: #e6fffb;
  }

  &.tag-member {
    color: #fa8c16;
    background: #fff7e6;
  }

  &.tag-alert {
    color: #faad14;
    background: #fffbe6;
  }

  &.tag-critical_alert {
    color: #f5222d;
    background: #fff1f0;
  }
}

.template-code {
  padding: 2px 6px;
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 12px;
  background: var(--el-fill-color-light);
  border-radius: 4px;
}

.template-pre {
  padding: 8px 10px;
  margin: 0;
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 12px;
  line-height: 1.6;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  background: var(--el-fill-color-light);
  border-radius: 4px;
}

.variable-list {
  display: flex;
  flex-direction: column;
  gap: 6px;

  .variable-item {
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 13px;

    code {
      padding: 1px 6px;
      font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
      font-size: 12px;
      color: var(--el-color-primary);
      background: var(--el-color-primary-light-9);
      border-radius: 3px;
    }

    .variable-desc {
      color: var(--el-text-color-secondary);
    }
  }
}
</style>
