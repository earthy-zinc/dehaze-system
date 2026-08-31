<!-- 保存为定时任务：常用频率快捷项 + 自定义 Cron + 下次触发时间预览，输入来源为固定文本，输出走站内消息 -->
<script lang="ts" setup>
import { ElMessage } from "element-plus";
import { AiScheduleAPI, type NextTimesPreview } from "dehaze-sdk-js";
import { watchDebounced } from "@vueuse/core";
import { computed, reactive, ref, watch } from "vue";
import { useChatStore } from "@/store/modules/chat";
import { useChatUserStore } from "@/store/modules/chatUser";

defineOptions({ name: "SaveAsTaskDialog" });

const visible = defineModel<boolean>({ default: false });

const chatStore = useChatStore();
const chatUserStore = useChatUserStore();

const CRON_PRESETS = [
  { label: "每天 09:00", cron: "0 9 * * *" },
  { label: "每天 18:00", cron: "0 18 * * *" },
  { label: "工作日 09:00", cron: "0 9 * * 1-5" },
  { label: "每周一 09:00", cron: "0 9 * * 1" },
];

const form = reactive({
  name: "",
  cron: "0 9 * * *",
  content: "",
});

const preview = ref<NextTimesPreview | null>(null);
const submitting = ref(false);

const currentConversation = computed(() =>
  chatStore.conversations.find(
    (item) => item.id === chatStore.currentConversationId
  )
);

const cronValid = computed(() => form.cron.trim().split(/\s+/).length === 5);

watch(visible, (value) => {
  if (!value) return;
  preview.value = null;
  form.name = currentConversation.value
    ? `${currentConversation.value.title}（定时执行）`
    : "定时对话任务";
  form.content = chatUserStore.getDraft(chatStore.currentConversationId);
  void refreshPreview();
});

// Cron 变更防抖刷新预览
watchDebounced(
  () => form.cron,
  () => {
    if (visible.value) void refreshPreview();
  },
  { debounce: 500 }
);

async function refreshPreview() {
  if (!cronValid.value) {
    preview.value = null;
    return;
  }
  preview.value = await AiScheduleAPI.previewNextTimes(form.cron.trim(), 5);
}

async function handleSubmit() {
  if (!form.name.trim()) {
    ElMessage.warning("请输入任务名称");
    return;
  }
  if (!cronValid.value) {
    ElMessage.warning("Cron 需为 5 位表达式，如 0 9 * * *");
    return;
  }
  submitting.value = true;
  try {
    await AiScheduleAPI.create({
      name: form.name.trim(),
      cron: form.cron.trim(),
      input: { type: "fixed", content: form.content.trim() },
      output: { type: "message" },
    });
    ElMessage.success("定时任务已创建，执行结果将通过站内消息通知");
    visible.value = false;
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <el-dialog
    v-model="visible"
    title="保存为定时任务"
    width="560px"
    :close-on-click-modal="false"
  >
    <el-form label-width="90px">
      <el-form-item label="任务名称">
        <el-input v-model="form.name" placeholder="任务名称" />
      </el-form-item>
      <el-form-item label="执行频率">
        <el-select v-model="form.cron" placeholder="选择常用频率或自定义">
          <el-option
            v-for="preset in CRON_PRESETS"
            :key="preset.cron"
            :label="preset.label"
            :value="preset.cron"
          />
        </el-select>
        <el-input
          v-model="form.cron"
          placeholder="或输入 5 位 Cron 表达式"
          class="mt-2"
        />
      </el-form-item>
      <el-form-item v-if="preview" label="下次触发">
        <div class="task-preview">
          <div class="task-preview__desc">{{ preview.description }}</div>
          <div
            v-for="time in preview.nextTimes"
            :key="time"
            class="task-preview__time"
          >
            {{ time.slice(0, 16).replace("T", " ") }}
          </div>
        </div>
      </el-form-item>
      <el-form-item label="执行内容">
        <el-input
          v-model="form.content"
          type="textarea"
          :rows="4"
          placeholder="定时触发时发送给 AI 的指令，默认取当前输入草稿"
        />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="visible = false">取消</el-button>
      <el-button type="primary" :loading="submitting" @click="handleSubmit">
        创建任务
      </el-button>
    </template>
  </el-dialog>
</template>

<style scoped lang="scss">
.task-preview {
  width: 100%;

  &__desc {
    margin-bottom: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__time {
    font-size: 13px;
    line-height: 1.8;
  }
}
</style>
