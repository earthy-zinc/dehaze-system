<template>
  <PageLayout level="L2" title="帮助中心" class="page">
    <view class="main-content">
      <view class="faq-list">
        <view
          v-for="(item, i) in faqs"
          :key="i"
          class="faq-card"
          @click="toggleFaq(i)"
        >
          <view class="faq-header">
            <text class="faq-q">{{ item.q }}</text>
            <SvgIcon
              :name="expandedIndex === i ? 'minus' : 'plus'"
              size="18"
              color="#3b82f6"
            />
          </view>
          <view v-if="expandedIndex === i" class="faq-body">
            <text class="faq-a">{{ item.a }}</text>
          </view>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";

const expandedIndex = ref(-1);

const faqs = [
  {
    q: "如何使用去雾功能？",
    a: "进入「去雾」Tab，上传图片后选择算法，调节参数即可开始处理。",
  },
  {
    q: "什么是会员权益？",
    a: "VIP 会员可获得更高的处理次数、批量处理、原图下载等高级功能。",
  },
  {
    q: "如何查看处理历史？",
    a: "在「我的」Tab 的个人数据分组中，点击「处理历史」即可查看。",
  },
  {
    q: "处理失败怎么办？",
    a: "处理失败后可在处理历史中找到对应记录，点击「重新处理」重试。",
  },
  {
    q: "如何提交反馈？",
    a: "在「我的」Tab 中选择「反馈评价」，点击提交反馈即可。",
  },
  {
    q: "如何修改个人信息？",
    a: "个人信息目前由系统统一管理，如需修改请联系管理员。",
  },
];

function toggleFaq(index: number) {
  expandedIndex.value = expandedIndex.value === index ? -1 : index;
}
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}
.main-content {
  padding: $spacing-md;
}

.faq-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.faq-card {
  background: #fff;
  border-radius: $radius-lg;
  padding: 24rpx;
  box-shadow: $shadow-sm;
  &:active {
    background: #f9fafb;
  }
}
.faq-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.faq-q {
  font-size: $font-md;
  font-weight: 500;
  color: $color-text-primary;
  flex: 1;
}
.faq-body {
  margin-top: 16rpx;
  padding-top: 16rpx;
  border-top: 1rpx solid $color-border-light;
}
.faq-a {
  font-size: $font-sm;
  color: $color-text-secondary;
  line-height: 1.8;
}
</style>
