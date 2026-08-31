<!-- 长期记忆面板：查看/搜索/手动录入/编辑/删除/导出，清空与恢复必须先二次确认再传 confirm=true -->
<script lang="ts" setup>
import { ElMessage, ElMessageBox } from "element-plus";
import {
  AiConversationAPI,
  type MemoryType,
  type MemoryVO,
} from "dehaze-sdk-js";
import { computed, reactive, ref, watch } from "vue";
import { downloadBlob } from "@/composables/useImportExport";

defineOptions({ name: "MemoryPanel" });

const visible = defineModel<boolean>({ default: false });

const PAGE_SIZE = 20;
const memoryTypeOptions: Array<{ label: string; value: MemoryType }> = [
  { label: "情景记忆", value: "episodic" },
  { label: "语义记忆", value: "semantic" },
  { label: "程序记忆", value: "procedural" },
];

const memories = ref<MemoryVO[]>([]);
const total = ref(0);
const loading = ref(false);
const keyword = ref("");
const memoryType = ref<MemoryType | "">("");
const pageNum = ref(1);

const typeLabel = (type: string) =>
  memoryTypeOptions.find((item) => item.value === type)?.label ?? type;

// 搜索接口命中即整页返回，无分页
const isSearchMode = computed(() => keyword.value.trim().length > 0);

async function load() {
  loading.value = true;
  try {
    if (isSearchMode.value) {
      memories.value = await AiConversationAPI.searchMemories(
        keyword.value.trim(),
        50
      );
      total.value = memories.value.length;
      return;
    }
    const result = await AiConversationAPI.getMemories({
      pageNum: pageNum.value,
      pageSize: PAGE_SIZE,
      memoryType: memoryType.value || undefined,
    });
    memories.value = result.list ?? [];
    total.value = result.total ?? 0;
  } finally {
    loading.value = false;
  }
}

watch(visible, (value) => {
  if (value) load();
});

watch([memoryType], () => {
  pageNum.value = 1;
  load();
});

function handleSearch() {
  pageNum.value = 1;
  load();
}

function handlePageChange(page: number) {
  pageNum.value = page;
  load();
}

// ===== 手动录入 =====
const createVisible = ref(false);
const createForm = reactive({
  memoryType: "semantic" as MemoryType,
  content: "",
  importance: 50,
});

async function handleCreate() {
  if (!createForm.content.trim()) {
    ElMessage.warning("请输入记忆内容");
    return;
  }
  await AiConversationAPI.createMemory({
    memoryType: createForm.memoryType,
    content: createForm.content.trim(),
    importance: createForm.importance,
    source: "manual",
  });
  ElMessage.success("记忆已录入");
  createVisible.value = false;
  createForm.content = "";
  createForm.importance = 50;
  load();
}

function handleEdit(memory: MemoryVO) {
  ElMessageBox.prompt("修改记忆内容", "编辑记忆", {
    inputValue: memory.content,
    inputType: "textarea",
  })
    .then(async ({ value }) => {
      if (!value?.trim()) return;
      await AiConversationAPI.updateMemory(memory.id, {
        content: value.trim(),
      });
      ElMessage.success("记忆已更新");
      load();
    })
    .catch(() => {});
}

function handleDelete(memory: MemoryVO) {
  ElMessageBox.confirm(
    "确认删除该条记忆？删除后 30 天内可通过恢复找回",
    "删除确认",
    {
      type: "warning",
    }
  )
    .then(async () => {
      await AiConversationAPI.deleteMemory(memory.id);
      ElMessage.success("记忆已删除");
      load();
    })
    .catch(() => {});
}

// clear/restore 的 confirm 参数走 query string，未确认传 false 会被后端 A0400 拒绝
function handleClear() {
  ElMessageBox.confirm(
    "确认清空全部长期记忆？清空后 30 天内可恢复",
    "清空记忆",
    { type: "warning" }
  )
    .then(async () => {
      const count = await AiConversationAPI.clearMemories(undefined, true);
      ElMessage.success(`已清空 ${count} 条记忆`);
      load();
    })
    .catch(() => {});
}

function handleRestore() {
  ElMessageBox.confirm("恢复 30 天内被清空/删除的记忆？", "恢复记忆", {
    type: "info",
  })
    .then(async () => {
      const count = await AiConversationAPI.restoreMemories(undefined, true);
      ElMessage.success(`已恢复 ${count} 条记忆`);
      load();
    })
    .catch(() => {});
}

async function handleExport(format: "json" | "markdown") {
  const blob = await AiConversationAPI.exportMemories(format);
  downloadBlob(blob, `memories.${format === "json" ? "json" : "md"}`);
  ElMessage.success("记忆已导出");
}
</script>

<template>
  <el-drawer v-model="visible" title="长期记忆" size="560px">
    <div class="memory-panel">
      <div class="memory-panel__toolbar">
        <el-input
          v-model="keyword"
          placeholder="搜索记忆内容"
          clearable
          style="width: 180px"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        />
        <el-select
          v-model="memoryType"
          placeholder="全部类型"
          clearable
          style="width: 120px"
        >
          <el-option
            v-for="option in memoryTypeOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
        <el-button type="primary" @click="createVisible = true"
          >手动录入</el-button
        >
        <el-dropdown trigger="click" @command="handleExport">
          <el-button>导出</el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="json">JSON</el-dropdown-item>
              <el-dropdown-item command="markdown">Markdown</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>

      <div class="memory-panel__list" v-loading="loading">
        <div v-for="memory in memories" :key="memory.id" class="memory-item">
          <div class="memory-item__content">{{ memory.content }}</div>
          <div class="memory-item__meta">
            <el-tag size="small">{{ typeLabel(memory.memoryType) }}</el-tag>
            <span>重要度 {{ memory.importance }}</span>
            <span v-if="memory.source === 'manual'">手动录入</span>
            <span>{{ memory.createTime?.slice(0, 10) }}</span>
          </div>
          <div class="memory-item__actions">
            <el-button link size="small" @click="handleEdit(memory)"
              >编辑</el-button
            >
            <el-button
              link
              size="small"
              type="danger"
              @click="handleDelete(memory)"
            >
              删除
            </el-button>
          </div>
        </div>
        <el-empty
          v-if="memories.length === 0 && !loading"
          description="暂无记忆，对话中确认的长期信息会自动沉淀到这里"
        />
      </div>

      <el-pagination
        v-if="!isSearchMode && total > PAGE_SIZE"
        layout="prev, pager, next"
        :total="total"
        :page-size="PAGE_SIZE"
        :current-page="pageNum"
        class="memory-panel__pagination"
        @current-change="handlePageChange"
      />

      <div class="memory-panel__danger">
        <el-button size="small" @click="handleRestore"
          >恢复已清空记忆</el-button
        >
        <el-button size="small" type="danger" @click="handleClear"
          >清空全部</el-button
        >
      </div>
    </div>

    <el-dialog
      v-model="createVisible"
      title="手动录入记忆"
      width="420px"
      append-to-body
    >
      <el-form label-width="80px">
        <el-form-item label="类型">
          <el-select v-model="createForm.memoryType">
            <el-option
              v-for="option in memoryTypeOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="内容">
          <el-input
            v-model="createForm.content"
            type="textarea"
            :rows="4"
            placeholder="例如：我在做毕业设计，主题是图像去雾"
          />
        </el-form-item>
        <el-form-item label="重要度">
          <el-slider v-model="createForm.importance" :max="100" show-input />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" @click="handleCreate">保存</el-button>
      </template>
    </el-dialog>
  </el-drawer>
</template>

<style scoped lang="scss">
.memory-panel {
  display: flex;
  flex-direction: column;
  height: 100%;

  &__toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 12px;
  }

  &__list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }

  &__pagination {
    justify-content: center;
    margin-top: 12px;
  }

  &__danger {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    padding-top: 12px;
    border-top: 1px solid var(--el-border-color-lighter);
  }
}

.memory-item {
  padding: 10px 12px;
  margin-bottom: 8px;
  background-color: var(--el-fill-color-lighter);
  border-radius: 6px;

  &__content {
    font-size: 14px;
    line-height: 1.6;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
  }

  &__meta {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 6px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__actions {
    margin-top: 4px;
    text-align: right;
  }
}
</style>
