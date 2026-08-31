<!-- 新会话开场引导：问候语 + 本地静态示例问题，点击即发送，零接口请求不消耗 Token -->
<script lang="ts" setup>
defineOptions({ name: "ChatEmptyState" });

const emit = defineEmits<{
  select: [question: string];
}>();

const sampleQuestions = [
  "帮我总结一份产品需求文档的写作框架",
  "用 Python 写一个快速排序，并解释时间复杂度",
  "推荐适合低照度图像增强的算法并对比优劣",
  "把这段话润色得更专业：我们做了个新功能，效果挺好的",
];
</script>

<template>
  <div class="empty-state">
    <h2 class="empty-state__greeting">你好，我是 Dehaze AI 助手</h2>
    <p class="empty-state__sub">
      可以问答、写作、写代码、分析图像，试试下面的问题
    </p>
    <div class="empty-state__samples">
      <el-button
        v-for="question in sampleQuestions"
        :key="question"
        plain
        class="empty-state__sample"
        @click="emit('select', question)"
      >
        {{ question }}
      </el-button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.empty-state {
  display: flex;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 0;
  padding: 24px;
  text-align: center;

  &__greeting {
    margin: 0 0 8px;
    font-size: 22px;
  }

  &__sub {
    margin: 0 0 24px;
    color: var(--el-text-color-secondary);
  }

  &__samples {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
    max-width: 480px;
  }

  &__sample {
    justify-content: flex-start;
    height: auto;
    padding: 10px 14px;
    text-align: left;
    white-space: normal;
  }
}
</style>
