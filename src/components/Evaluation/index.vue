<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";

defineProps({
  // 评价指标名称
  name: {
    type: String,
    default: "峰值信噪比（PSNR）",
  },
  value: {
    type: Number,
    default: 1,
  },
  multiple: {
    type: Number,
    default: 1,
  },
});
</script>

<template>
  <div class="ev-card-wrap">
    <div class="left-wrap">
      <div class="left-top-wrap">{{ name }}</div>
      <div class="left-middle-wrap">{{ value }}</div>
      <div class="left-bottom-wrap">⬆ {{ multiple }}倍</div>
    </div>
    <div class="right-wrap">
      <SvgIcon
        iconClass="up_quxian"
        color="var(--el-color-primary)"
        size="100px"
      />
    </div>
  </div>
</template>

<style lang="scss" scoped>
.ev-card-wrap {
  display: flex;
  width: 100%;
  padding: 20px;
  border: 1px solid #f3f3f3;
  border-radius: 20px;

  &:hover {
    color: #fcfcfc;
    background: linear-gradient(
      35.3deg,
      var(--el-color-primary-light-7) 0,
      var(--el-color-primary) 100%
    );
  }

  .left-wrap {
    width: 60%;

    .left-top-wrap {
      margin-top: 5px;
    }

    .left-middle-wrap {
      margin: 10px 0;
      font-size: 24px;
      font-weight: 700;
    }

    .left-bottom-wrap {
      margin: 5px 0;
      font-weight: 700;
      color: var(--el-color-primary);
    }
  }

  .right-wrap {
    width: 40%;
  }
}
</style>
