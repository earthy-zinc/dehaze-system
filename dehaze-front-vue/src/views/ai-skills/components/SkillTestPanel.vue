<!-- Skill 试运行：输入测试数据预览指令执行效果，不入库不推送、不触发真实 LLM 推理 -->
<script lang="ts" setup>
import { useAdminSkillStore } from "@/store/modules/adminSkill";

defineOptions({ name: "SkillTestPanel" });

const skillStore = useAdminSkillStore();
const inputText = ref("");

const skill = computed(() => skillStore.testDialog.skill);
const instruction = computed(() =>
  String(skillStore.testResult?.instruction ?? "")
);
const inputPreview = computed(() =>
  skillStore.testResult
    ? JSON.stringify(skillStore.testResult.input ?? null, null, 2)
    : ""
);

watch(
  () => skillStore.testDialog.visible,
  (visible) => {
    if (visible) inputText.value = "";
  }
);

async function run() {
  const current = skill.value;
  if (!current) return;
  const text = inputText.value.trim();
  // 测试输入可填 JSON 结构，也可直接填原始文本
  let inputData: unknown = text;
  if (text) {
    try {
      inputData = JSON.parse(text);
    } catch {
      inputData = text;
    }
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
      title="试运行仅构造指令与输入预览，不入库、不推送、不触发真实 LLM 推理"
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

    <template v-if="skillStore.testResult">
      <el-divider content-position="left">预览结果</el-divider>
      <el-collapse>
        <el-collapse-item title="指令预览" name="instruction">
          <pre class="preview-block">{{ instruction }}</pre>
        </el-collapse-item>
        <el-collapse-item title="输入预览" name="input">
          <pre class="preview-block">{{ inputPreview }}</pre>
        </el-collapse-item>
      </el-collapse>
    </template>

    <template #footer>
      <el-button @click="skillStore.testDialog.visible = false">关闭</el-button>
      <el-button
        v-hasPerm="['ai:skill:manage']"
        type="primary"
        :loading="skillStore.testLoading"
        :disabled="skill?.status !== 1"
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
