<!-- 会话设置弹窗：标题/模型/智能体/系统提示词，会话级配置收敛于弹窗，变更下一条消息生效 -->
<script lang="ts" setup>
import { ElMessage } from "element-plus";
import { AiAgentAPI, type AgentListItem } from "dehaze-sdk-js";
import { computed, reactive, ref, watch } from "vue";
import { useChatStore } from "@/store/modules/chat";
import { useChatUserStore } from "@/store/modules/chatUser";

defineOptions({ name: "ConversationSettings" });

const visible = defineModel<boolean>({ default: false });

const chatStore = useChatStore();
const chatUserStore = useChatUserStore();

// 表单本地字段均可空输入，提交时转换为 ConversationUpdateForm 的可选字段
const form = reactive({
  title: "",
  model: "",
  agentCode: "",
  systemPrompt: "",
  suggestionsEnabled: true,
});

const currentConversation = computed(() =>
  chatStore.conversations.find(
    (item) => item.id === chatStore.currentConversationId
  )
);

const enabledAgents = ref<AgentListItem[]>([]);

watch(visible, (value) => {
  if (!value) return;
  form.title = currentConversation.value?.title ?? "";
  form.model = currentConversation.value?.model ?? "";
  form.agentCode = currentConversation.value?.agentCode ?? "";
  form.systemPrompt = currentConversation.value?.systemPrompt ?? "";
  form.suggestionsEnabled =
    (currentConversation.value?.suggestionsEnabled ?? 1) === 1;
  AiAgentAPI.listEnabled()
    .then((agents) => (enabledAgents.value = agents))
    .catch(() => {});
});

async function handleSave() {
  if (!currentConversation.value) return;
  await chatStore.updateConversation(currentConversation.value.id, {
    title: form.title.trim() || undefined,
    model: form.model || undefined,
    agentCode: form.agentCode.trim() || undefined,
    systemPrompt: form.systemPrompt || undefined,
    suggestionsEnabled: form.suggestionsEnabled,
  });
  ElMessage.success("会话设置已保存，下一条消息生效");
  visible.value = false;
}
</script>

<template>
  <el-dialog
    v-model="visible"
    title="会话设置"
    width="520px"
    :close-on-click-modal="false"
  >
    <el-form label-width="90px">
      <el-form-item label="会话标题">
        <el-input v-model="form.title" placeholder="会话标题" />
      </el-form-item>
      <el-form-item label="模型">
        <el-select v-model="form.model" placeholder="保持当前模型" clearable>
          <el-option
            v-for="model in chatUserStore.availableModels"
            :key="model.modelId"
            :label="model.displayName"
            :value="model.modelId"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="智能体">
        <el-select
          v-model="form.agentCode"
          placeholder="默认智能体"
          clearable
          filterable
        >
          <el-option
            v-for="agent in enabledAgents"
            :key="agent.agentCode"
            :label="agent.name"
            :value="agent.agentCode"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="系统提示词">
        <el-input
          v-model="form.systemPrompt"
          type="textarea"
          :rows="4"
          placeholder="设定助手的角色与行为约束"
        />
      </el-form-item>
      <el-form-item label="推荐">
        <el-switch v-model="form.suggestionsEnabled" />
        <span class="ml-2 text-xs text-gray-400">
          回复后推荐类似问题，便于继续追问
        </span>
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="visible = false">取消</el-button>
      <el-button
        type="primary"
        :disabled="!currentConversation"
        @click="handleSave"
      >
        保存
      </el-button>
    </template>
  </el-dialog>
</template>
