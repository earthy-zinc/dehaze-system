<template>
  <el-dialog
    :model-value="visible"
    :show-close="false"
    :close-on-click-modal="false"
    title="算法效果评价"
    width="600px"
    class="rating-card-dialog"
    @update:model-value="(val: boolean) => emit('update:visible', val)"
  >
    <div class="algorithm-tip">
      <el-icon><Picture /></el-icon>
      <span>{{ algorithmName }}</span>
    </div>

    <el-form
      ref="formRef"
      :model="formData"
      :rules="formRules"
      label-width="90px"
    >
      <el-form-item label="整体评分" prop="rating">
        <el-rate
          v-model="formData.rating"
          :max="5"
          show-text
          :texts="['很不满意', '不满意', '一般', '满意', '非常满意']"
        />
      </el-form-item>

      <el-form-item v-if="availableTags.length" label="效果标签">
        <div class="tag-selector">
          <el-check-tag
            v-for="tag in availableTags"
            :key="tag"
            :checked="formData.tags.includes(tag)"
            @change="handleTagToggle(tag, $event)"
          >
            {{ tag }}
          </el-check-tag>
          <span class="tag-count-hint">
            最多选择 5 个（{{ formData.tags.length }}/5）
          </span>
        </div>
      </el-form-item>

      <el-form-item label="评价内容" prop="comment">
        <el-input
          v-model="formData.comment"
          type="textarea"
          :rows="4"
          maxlength="500"
          show-word-limit
          placeholder="说说您的使用体验（选填）"
        />
      </el-form-item>

      <el-form-item label="上传截图">
        <el-upload
          :file-list="fileList"
          :http-request="handleUpload"
          :before-upload="handleBeforeUpload"
          list-type="picture-card"
          :limit="3"
          :on-exceed="handleExceed"
          :on-remove="handleRemove"
          accept="image/jpeg,image/png,image/webp"
        >
          <el-icon class="upload-icon"><Plus /></el-icon>
        </el-upload>
        <div class="upload-tip">
          支持 JPG/PNG/WEBP，单张不超过 5MB，最多 3 张
        </div>
      </el-form-item>

      <el-form-item label="匿名评价">
        <el-switch v-model="formData.isAnonymous" />
      </el-form-item>
    </el-form>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="handleSkip">跳过</el-button>
        <el-button
          type="primary"
          :loading="submitLoading"
          @click="handleSubmit"
        >
          提交评价
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script lang="ts" setup>
import { FeedbackAPI, FileAPI } from "dehaze-sdk-js";
import { Plus, Picture } from "@element-plus/icons-vue";
import type {
  UploadFile,
  UploadFiles,
  UploadRawFile,
  UploadRequestOptions,
  UploadUserFile,
} from "element-plus";

defineOptions({ name: "RatingCard" });

const props = defineProps<{
  predLogId: number;
  algorithmName: string;
  visible: boolean;
}>();

const emit = defineEmits<{
  (e: "update:visible", val: boolean): void;
  (e: "success"): void;
}>();

const POSITIVE_TAGS = [
  "去雾彻底",
  "色彩自然",
  "细节清晰",
  "处理速度快",
  "整体提升明显",
];
const NEGATIVE_TAGS = [
  "残留雾气",
  "色彩失真",
  "细节丢失",
  "处理速度慢",
  "无明显改善",
];

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_SIZE = 5 * 1024 * 1024;
const MAX_TAGS = 5;

const formRef = ref(ElForm);
const submitLoading = ref(false);
const fileList = ref<UploadUserFile[]>([]);
const imageUrls = ref<string[]>([]);

const formData = reactive({
  rating: 0,
  comment: "",
  tags: [] as string[],
  isAnonymous: false,
});

const formRules = reactive({
  rating: [
    { required: true, message: "请选择评分", trigger: "change" },
    {
      validator: (_rule: any, value: number, callback: any) => {
        if (value < 1 || value > 5) {
          callback(new Error("评分范围为 1-5"));
        } else {
          callback();
        }
      },
      trigger: "change",
    },
  ],
});

const availableTags = computed<string[]>(() => {
  const rating = formData.rating;
  if (rating === 0) return [];
  if (rating >= 4) return POSITIVE_TAGS;
  if (rating <= 2) return NEGATIVE_TAGS;
  return [...POSITIVE_TAGS, ...NEGATIVE_TAGS];
});

watch(
  () => formData.rating,
  (newRating, oldRating) => {
    const wasPositive = oldRating >= 4;
    const wasNegative = oldRating <= 2;
    const wasNeutral = oldRating === 3;
    const isPositive = newRating >= 4;
    const isNegative = newRating <= 2;
    const isNeutral = newRating === 3;

    const changedCategory =
      (wasPositive && !isPositive) ||
      (wasNegative && !isNegative) ||
      (wasNeutral && !isNeutral && newRating !== 0);

    if (changedCategory) {
      formData.tags = [];
    }
  }
);

watch(
  () => props.visible,
  (val) => {
    if (val) {
      formData.rating = 0;
      formData.comment = "";
      formData.tags = [];
      formData.isAnonymous = false;
      fileList.value = [];
      imageUrls.value = [];
      formRef.value?.clearValidate();
    }
  }
);

function handleTagToggle(tag: string, checked: boolean) {
  if (checked) {
    if (formData.tags.length >= MAX_TAGS) {
      ElMessage.warning(`最多选择 ${MAX_TAGS} 个标签`);
      return;
    }
    if (!formData.tags.includes(tag)) {
      formData.tags = [...formData.tags, tag];
    }
  } else {
    formData.tags = formData.tags.filter((t) => t !== tag);
  }
}

function handleBeforeUpload(file: UploadRawFile): boolean {
  if (!ALLOWED_TYPES.includes(file.type)) {
    ElMessage.error("仅支持 JPG/PNG/WEBP 格式");
    return false;
  }
  if (file.size > MAX_FILE_SIZE) {
    ElMessage.error("图片大小不能超过 5MB");
    return false;
  }
  return true;
}

async function handleUpload(options: UploadRequestOptions) {
  try {
    const data = await FileAPI.upload(options.file);
    imageUrls.value.push(data.url);
    fileList.value = [
      ...fileList.value,
      {
        name: options.file.name,
        url: data.url,
        status: "success",
      } as UploadUserFile,
    ];
  } catch (err: any) {
    ElMessage.error("图片上传失败：" + (err?.message || "未知错误"));
  }
}

function handleExceed() {
  ElMessage.warning("最多上传 3 张图片");
}

function handleRemove(file: UploadFile, files: UploadFiles) {
  const removedUrl = (file as UploadUserFile).url;
  if (removedUrl) {
    imageUrls.value = imageUrls.value.filter((url) => url !== removedUrl);
  }
  fileList.value = files;
}

function closeDialog() {
  emit("update:visible", false);
}

function handleSkip() {
  closeDialog();
}

const handleSubmit = useThrottleFn(() => {
  formRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    submitLoading.value = true;
    FeedbackAPI.createRating({
      predLogId: props.predLogId,
      rating: formData.rating,
      comment: formData.comment || undefined,
      tags: formData.tags.length ? formData.tags : undefined,
      imageUrls: imageUrls.value.length ? imageUrls.value : undefined,
      isAnonymous: formData.isAnonymous ? 1 : 0,
    })
      .then(() => {
        ElMessage.success("评价成功，获得成长值奖励");
        emit("success");
        closeDialog();
      })
      .finally(() => {
        submitLoading.value = false;
      });
  });
}, 3000);
</script>

<style lang="scss" scoped>
.rating-card-dialog {
  :deep(.el-dialog__body) {
    padding: 16px 20px;
  }
}

.algorithm-tip {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  padding: 6px 12px;
  margin-bottom: 16px;
  font-size: 13px;
  font-weight: 500;
  color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
  border-radius: 6px;
}

.tag-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;

  .tag-count-hint {
    margin-left: 8px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.upload-icon {
  font-size: 24px;
  color: var(--el-text-color-placeholder);
}

.upload-tip {
  margin-top: 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.dialog-footer {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}
</style>
