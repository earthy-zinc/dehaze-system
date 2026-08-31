<!--
  SKILL 表单对话框（Agent Skills 规范）：
  - 新建 = zip 压缩包上传（SKILL.md + reference/script/assets，后端解析 frontmatter 校验）
  - 编辑 = 文本表单（name/description/指令/状态）+ frontmatter 只读 + 资源文件清单（预览/下载）
-->
<script lang="ts" setup>
import type { FormInstance, UploadFile } from "element-plus";
import { SkillFileVO, SkillForm, SkillVO } from "dehaze-sdk-js";
import { Download } from "@element-plus/icons-vue";
import { useAdminSkillStore } from "@/store/modules/adminSkill";

defineOptions({ name: "SkillFormDialog" });

const emit = defineEmits<{ saved: [skill: SkillVO] }>();

// 指令内容上限（100KB）与危险操作正则，与后端 ai_skill 校验规则保持一致
const INSTRUCTION_MAX_BYTES = 100 * 1024;
const DANGEROUS_PATTERN =
  /(rm\s+-rf\s*\/|mkfs\.?(ext\d?|xfs|vfat|ntfs)?\b|curl[^\n]*\|\s*(ba)?sh\b|wget[^\n]*\|\s*(ba)?sh\b|sudo\s+(rm|shutdown|reboot|mkfs|dd)|dd\s+if=.*of=\/dev\/)/i;

const skillStore = useAdminSkillStore();
const formRef = ref<FormInstance>();

/** 编辑模式 = 已打开既有 Skill；创建模式 = 上传 zip */
const isEdit = computed(() => !!skillStore.skillForm.skill);

/** 编辑详情（含 frontmatter 字段与文件清单，列表不含这些字段需拉详情） */
const detail = ref<SkillVO | null>(null);
/** 创建模式选中的 zip 文件 */
const selectedFile = ref<File | null>(null);
const previewName = ref("");

const emptyForm = (): SkillForm => ({
  name: "",
  description: "",
  scene: "",
  instruction: "",
  status: 1,
});
const form = reactive<SkillForm>(emptyForm());

const rules = {
  name: [
    { required: true, message: "Skill 名称不能为空", trigger: "blur" },
    { max: 128, message: "名称不超过 128 个字符", trigger: "blur" },
  ],
  description: [{ max: 500, message: "描述不超过 500 个字符", trigger: "blur" }],
  instruction: [
    { required: true, message: "Markdown 指令不能为空", trigger: "blur" },
  ],
};

const instructionSize = computed(
  () => new TextEncoder().encode(form.instruction).length
);
const oversized = computed(() => instructionSize.value > INSTRUCTION_MAX_BYTES);
const dangerHit = computed(() => DANGEROUS_PATTERN.test(form.instruction));

watch(
  () => skillStore.skillForm.visible,
  async (visible) => {
    if (!visible) return;
    selectedFile.value = null;
    previewName.value = "";
    detail.value = null;
    Object.assign(form, emptyForm());
    const skill = skillStore.skillForm.skill;
    if (skill) {
      // 编辑模式：拉详情获取 frontmatter 与文件清单
      try {
        detail.value = await skillStore.fetchSkillDetail(skill.id);
      } catch {
        detail.value = skill;
      }
      const d = detail.value;
      Object.assign(form, {
        name: d.name,
        description: d.description ?? "",
        scene: d.scene ?? "",
        instruction: d.instruction ?? "",
        status: d.status,
      });
    }
  }
);

function handleFileChange(uploadFile: UploadFile) {
  const raw = uploadFile.raw;
  if (!raw) return;
  selectedFile.value = raw;
  previewName.value = raw.name;
}

function handleFileRemove() {
  selectedFile.value = null;
  previewName.value = "";
}

async function submit() {
  if (isEdit.value) {
    await submitEdit();
  } else {
    await submitCreate();
  }
}

async function submitCreate() {
  if (!selectedFile.value) {
    ElMessage.error("请选择 SKILL 压缩包（zip）");
    return;
  }
  const saved = await skillStore.uploadSkill(selectedFile.value);
  skillStore.skillForm.visible = false;
  ElMessage.success("SKILL 已上传并解析入库");
  emit("saved", saved);
}

async function submitEdit() {
  await formRef.value?.validate();
  if (oversized.value) {
    ElMessage.error(
      `指令内容 ${(instructionSize.value / 1024).toFixed(1)}KB，超过 100KB 上限`
    );
    return;
  }
  if (dangerHit.value) {
    ElMessage.error(
      "指令包含危险操作（如 rm -rf /、curl | bash、dd 写设备），禁止保存"
    );
    return;
  }
  const saved = await skillStore.saveSkill({ ...form });
  skillStore.skillForm.visible = false;
  ElMessage.success("Skill 已保存");
  emit("saved", saved);
}

/** 资源文件预览/下载（从对象存储读取） */
async function previewFile(file: SkillFileVO) {
  const id = detail.value?.id;
  if (!id) return;
  try {
    const blob = await skillStore.getSkillFile(id, file.path);
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank");
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } catch {
    ElMessage.error(`文件 ${file.path} 读取失败`);
  }
}

function formatSize(bytes?: number) {
  if (!bytes) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}
</script>

<template>
  <el-dialog
    v-model="skillStore.skillForm.visible"
    :title="isEdit ? '编辑 Skill' : '上传 SKILL'"
    width="720px"
    destroy-on-close
    append-to-body
  >
    <!-- ==================== 创建：zip 上传 ==================== -->
    <template v-if="!isEdit">
      <el-alert
        class="mb-3"
        type="info"
        :closable="false"
        title="SKILL 遵循 Agent Skills 规范：目录内须包含 SKILL.md（YAML frontmatter + 指令正文），可选 reference/ script/ assets/ 资源文件"
      />
      <el-upload
        drag
        action="#"
        :auto-upload="false"
        :limit="1"
        accept=".zip"
        :on-change="handleFileChange"
        :on-remove="handleFileRemove"
        :on-exceed="() => ElMessage.warning('每次仅支持上传一个 zip')"
      >
        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
        <div class="el-upload__text">拖拽 SKILL 压缩包到此处，或<em>点击选择</em></div>
        <template #tip>
          <div class="el-upload__tip">
            仅支持 .zip；上传后自动解析 SKILL.md frontmatter 并校验（name 命名规范、目录名一致、description）
          </div>
        </template>
      </el-upload>
      <div v-if="selectedFile" class="mt-2 flex items-center gap-2 text-sm">
        <el-tag type="success">已选择</el-tag>
        <span class="text-gray-600">{{ previewName }}</span>
        <span class="text-xs text-gray-400">
          {{ formatSize(selectedFile.size) }}
        </span>
      </div>
      <el-form label-width="110px" class="mt-4">
        <el-form-item label="状态">
          <el-switch v-model="form.status" :active-value="1" :inactive-value="0" />
          <span class="ml-2 text-xs text-gray-400">
            上传即启用；禁用后 LLM 不再自动选择
          </span>
        </el-form-item>
      </el-form>
    </template>

    <!-- ==================== 编辑：文本表单 + 详情 ==================== -->
    <template v-else>
      <el-descriptions v-if="detail" :column="2" border size="small" class="mb-3">
        <el-descriptions-item label="License">
          {{ detail.license ?? "-" }}
        </el-descriptions-item>
        <el-descriptions-item label="Compatibility">
          {{ detail.compatibility ?? "-" }}
        </el-descriptions-item>
        <el-descriptions-item label="Allowed Tools">
          {{ detail.allowedTools ?? "-" }}
        </el-descriptions-item>
        <el-descriptions-item label="资源文件">
          <span v-if="!detail.files?.length">-</span>
          <ul v-else class="file-list">
            <li v-for="f in detail.files" :key="f.path">
              <span class="file-path">{{ f.path }}</span>
              <span class="text-xs text-gray-400">{{ formatSize(f.fileSize) }}</span>
              <el-button link type="primary" size="small" @click="previewFile(f)">
                <el-icon class="mr-0.5"><Download /></el-icon>预览
              </el-button>
            </li>
          </ul>
        </el-descriptions-item>
      </el-descriptions>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="110px">
        <el-form-item label="名称" prop="name">
          <el-input v-model="form.name" placeholder="Skill 名称（唯一）" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input
            v-model="form.description"
            type="textarea"
            :rows="2"
            placeholder="一句话说明 Skill 用途"
          />
        </el-form-item>
        <el-form-item label="适用场景">
          <el-input v-model="form.scene" placeholder="如：图像去雾、报告生成" />
        </el-form-item>
        <el-form-item label="Markdown 指令" prop="instruction">
          <el-input
            v-model="form.instruction"
            class="instruction-input"
            type="textarea"
            :rows="10"
            placeholder="# 工作流标题&#10;1. 步骤一&#10;2. 步骤二"
          />
        </el-form-item>
        <el-form-item>
          <div class="w-full text-xs text-gray-400">
            已输入 {{ (instructionSize / 1024).toFixed(1) }}KB / 上限 100KB；
            会话启动时仅加载名称与描述，命中后才加载完整指令
          </div>
          <el-alert
            v-if="oversized"
            class="mt-1"
            type="error"
            :closable="false"
            title="指令内容超过 100KB 上限"
          />
          <el-alert
            v-else-if="dangerHit"
            class="mt-1"
            type="error"
            :closable="false"
            title="指令包含危险操作（rm -rf /、curl | bash、mkfs、dd 写设备等）"
          />
        </el-form-item>
        <el-form-item label="状态">
          <el-switch v-model="form.status" :active-value="1" :inactive-value="0" />
          <span class="ml-2 text-xs text-gray-400">
            禁用后 LLM 不再自动选择，进行中的对话不受影响
          </span>
        </el-form-item>
      </el-form>
    </template>

    <template #footer>
      <el-button @click="skillStore.skillForm.visible = false">取消</el-button>
      <el-button
        v-hasPerm="['ai:skill:manage']"
        type="primary"
        :loading="skillStore.submitting"
        @click="submit"
      >
        {{ isEdit ? "确定" : "上传并启用" }}
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.instruction-input :deep(textarea) {
  font-family: Menlo, Consolas, monospace;
}

.file-list {
  margin: 0;
  padding: 0;
  list-style: none;
  max-height: 120px;
  overflow-y: auto;

  li {
    display: flex;
    align-items: center;
    gap: 8px;

    .file-path {
      font-family: Menlo, Consolas, monospace;
      font-size: 12px;
    }
  }
}
</style>
