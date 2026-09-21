<!-- Skill 试运行：以 Skill 指令为系统提示词真实推理一次，独立调试会话不入库不推送 -->
<script lang="ts" setup>
import { useAdminSkillStore } from "@/store/modules/adminSkill";

defineOptions({ name: "SkillTestPanel" });

const skillStore = useAdminSkillStore();
const inputText = ref("");

const skill = computed(() => skillStore.testDialog.skill);
const result = computed(() => skillStore.testResult);
const instruction = computed(
  () => result.value?.instruction ?? skill.value?.instruction ?? ""
);
const inputPreview = computed(() =>
  result.value ? JSON.stringify(result.value.input ?? null, null, 2) : ""
);
const usageText = computed(() =>
  JSON.stringify(result.value?.usage ?? {}, null, 2)
);

watch(
  () => skillStore.testDialog.visible,
  (visible) => {
    if (visible) inputText.value = "";
  }
);

async function run() {
  const current = skill.value;
  const text = inputText.value.trim();
  if (!current || !text) return;
  // 测试输入可填 JSON 结构，也可直接填原始文本
  let inputData: unknown;
  try {
    inputData = JSON.parse(text);
  } catch {
    inputData = text;
  }
  await skillStore.testSkill(current.id, inputData);
}
</script>

<template>
  <el-dialog
    v-model="skillStore.testDialog.visible"
    :title="`试运行 - ${skill?.name ?? ''}`"
    width="720px"
    destroy-on-close
    append-to-body
  >
    <el-alert
      class="mb-3"
      type="info"
      :closable="false"
      title="试运行以该 Skill 指令作为系统提示词真实调用一次模型推理（独立调试会话，不入库、不推送、不污染生产会话）"
    />
    <el-alert
      v-if="skill && skill.status !== 1"
      class="mb-3"
      type="warning"
      :closable="false"
      title="该 Skill 当前为禁用状态，请先启用后再试运行"
    />

    <el-form label-width="90px">
      <el-form-item label="测试数据">
        <el-input
          v-model="inputText"
          type="textarea"
          :rows="4"
          placeholder='支持 JSON 结构，如 {"imageUrl": "https://..."}，或直接输入文本'
        />
      </el-form-item>
    </el-form>

    <template v-if="result">
      <el-divider content-position="left">推理结果</el-divider>
      <el-form label-width="90px">
        <el-form-item label="模型输出">
          <pre class="preview-block">{{
            result.output || "（未返回内容）"
          }}</pre>
        </el-form-item>
      </el-form>
      <el-collapse>
        <el-collapse-item title="指令（本次系统提示词）" name="instruction">
          <pre class="preview-block">{{ instruction }}</pre>
        </el-collapse-item>
        <el-collapse-item title="输入预览" name="input">
          <pre class="preview-block">{{ inputPreview }}</pre>
        </el-collapse-item>
        <el-collapse-item title="用量" name="usage">
          <pre class="preview-block">{{ usageText }}</pre>
        </el-collapse-item>
      </el-collapse>
    </template>

    <template #footer>
      <el-button @click="skillStore.testDialog.visible = false">关闭</el-button>
      <el-button
        v-hasPerm="['ai:skill:manage']"
        type="primary"
        :loading="skillStore.testLoading"
        :disabled="skill?.status !== 1 || !inputText.trim()"
        @click="run"
      >
        运行
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.preview-block {
  max-height: 320px;
  padding: 8px;
  overflow: auto;
  word-break: break-all;
  white-space: pre-wrap;
  background: #f5f7fa;
  border-radius: 4px;
}
</style>
