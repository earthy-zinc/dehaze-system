<!-- 渐进式反馈栏：点赞一键完成；点踩展开问题标签单选 + 可选说明；已反馈高亮，支持撤销/修改 -->
<script lang="ts" setup>
import type { FeedbackForm, FeedbackVO } from "dehaze-sdk-js";
import { computed, ref } from "vue";

defineOptions({ name: "FeedbackBar" });

const props = defineProps<{
  messageId: number;
  feedback?: FeedbackVO | null;
}>();

const emit = defineEmits<{
  submit: [data: FeedbackForm];
  cancel: [];
}>();

const LIKE_TAGS = ["accurate", "detailed", "concise", "creative"];
const DISLIKE_TAGS = [
  { value: "incorrect", label: "内容错误" },
  { value: "irrelevant", label: "答非所问" },
  { value: "incomplete", label: "内容不完整" },
  { value: "too_long", label: "过于冗长" },
  { value: "bad_citation", label: "引用错误" },
  { value: "harmful", label: "有害内容" },
];

const expanded = ref(false);
const dislikeTag = ref<string>("");
const comment = ref("");

const liked = computed(() => props.feedback?.rating === 1);
const disliked = computed(() => props.feedback?.rating === -1);

function like() {
  emit("submit", { rating: 1 });
}

function expandDislike() {
  expanded.value = true;
  dislikeTag.value = props.feedback?.tags?.[0] ?? "";
  comment.value = props.feedback?.comment ?? "";
}

function submitDislike() {
  emit("submit", {
    rating: -1,
    tags: dislikeTag.value ? [dislikeTag.value] : undefined,
    comment: comment.value.trim() || undefined,
  });
}
</script>

<template>
  <div class="feedback-bar">
    <div class="feedback-bar__quick">
      <el-button
        size="small"
        :type="liked ? 'primary' : 'default'"
        :plain="!liked"
        @click="like"
      >
        👍 有帮助
      </el-button>
      <el-button
        size="small"
        :type="disliked ? 'danger' : 'default'"
        :plain="!disliked"
        @click="expandDislike"
      >
        👎 待改进
      </el-button>
      <el-button v-if="feedback" size="small" link @click="emit('cancel')"
        >撤销反馈</el-button
      >
      <el-button v-else size="small" link @click="emit('cancel')"
        >收起</el-button
      >
    </div>

    <div v-if="expanded" class="feedback-bar__detail">
      <div class="feedback-bar__tags">
        <el-radio-group v-model="dislikeTag">
          <el-radio-button
            v-for="tag in DISLIKE_TAGS"
            :key="tag.value"
            :value="tag.value"
          >
            {{ tag.label }}
          </el-radio-button>
        </el-radio-group>
      </div>
      <el-input
        v-model="comment"
        type="textarea"
        :rows="2"
        maxlength="500"
        placeholder="补充改进建议（可选）"
      />
      <div class="feedback-bar__actions">
        <el-button type="primary" size="small" @click="submitDislike"
          >提交</el-button
        >
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.feedback-bar {
  max-width: 92%;
  padding: 8px 10px;
  margin-top: 4px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;

  &__quick {
    display: flex;
    gap: 8px;
  }

  &__detail {
    margin-top: 8px;
  }

  &__tags {
    margin-bottom: 8px;
  }

  &__actions {
    margin-top: 8px;
    text-align: right;
  }
}
</style>
