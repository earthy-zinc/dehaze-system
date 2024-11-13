<script lang="ts" setup>
import { ref } from "vue";
import { ViewCard } from "../Waterfall/types";
import { useCalculateCols, useLayout } from "../Waterfall/waterfall";
import { LazyType } from "../LazyImg/types";
import Lazy from "../LazyImg/lazy";
import { getValue } from "@/utils";
import { api as viewerApi } from "v-viewer";

defineOptions({
  name: "LongitudinalWaterfall",
});

const props = defineProps({
  list: {
    type: Array as PropType<ViewCard[]>,
    default: () => [],
  },
  rowKey: {
    type: String,
    default: "id",
  },
  imgSelector: {
    type: String,
    default: "src",
  },
  // 宽度固定为 200
  width: {
    type: Number,
    default: 300,
  },
  breakpoints: {
    type: Object,
    default: () => ({
      1200: {
        // when wrapper width < 1200
        rowPerView: 3,
      },
      800: {
        // when wrapper width < 800
        rowPerView: 2,
      },
      500: {
        // when wrapper width < 500
        rowPerView: 1,
      },
    }),
  },
  gutter: {
    type: Number,
    default: 10,
  },
  hasAroundGutter: {
    type: Boolean,
    default: true,
  },
  posDuration: {
    type: Number,
    default: 300,
  },
  animationPrefix: {
    type: String,
    default: "animate__animated",
  },
  animationEffect: {
    type: String,
    default: "fadeIn",
  },
  animationDuration: {
    type: Number,
    default: 1000,
  },
  animationDelay: {
    type: Number,
    default: 300,
  },
  backgroundColor: {
    type: String,
    default: "#fff",
  },
  lazyload: {
    type: Boolean,
    default: true,
  },
  loadProps: {
    type: Object,
    default: () => {},
  },
  crossOrigin: {
    type: Boolean,
    default: true,
  },
  delay: {
    type: Number,
    default: 300,
  },
  align: {
    type: String,
    default: "center",
  },
  speed: {
    type: Number,
    default: 1,
  },
});

const emit = defineEmits(["afterRender"]);

const imageRef = ref<HTMLImageElement>();

const lazy: LazyType = new Lazy(
  props.lazyload,
  props.loadProps,
  props.crossOrigin
);
provide("lazy", lazy);

// 容器块信息
const { waterfallWrapper, wrapperWidth, colWidth, cols, offsetX } =
  useCalculateCols(props);

// 容器高度，块定位
const { layoutHandle } = useLayout(
  props,
  colWidth,
  cols,
  offsetX,
  waterfallWrapper
);

// 1s内最多执行一次排版，减少性能开销
const renderer = useDebounceFn(() => {
  layoutHandle().then(() => {
    emit("afterRender");
  });
}, props.delay);

// 列表发生变化直接触发排版
watch(
  () => [wrapperWidth, colWidth, props.list],
  () => {
    if (wrapperWidth.value > 0) renderer();
  },
  { deep: true }
);

// 尺寸宽度变化防抖触发
const sizeChangeTime = ref(0);

watchDebounced(
  colWidth,
  () => {
    layoutHandle();
    sizeChangeTime.value += 1;
  },
  { debounce: props.delay }
);

// 图片加载完成
provide("imgLoaded", renderer);

// 根据选择器获取图片地址
const getRenderURL = (item: ViewCard): string => {
  return getValue(item, props.imgSelector)[0];
};

// 获取唯一值
const getKey = (item: ViewCard, index: number): string => {
  return item[props.rowKey] || index;
};

// 点击图片展示高清大图
const showBigPicture = (index: number) => {
  viewerApi({
    images: props.list.map((item) => getRenderURL(item)),
  })
    .show()
    .update()
    .view(index);
};
</script>

<template>
  <div ref="waterfallWrapper" class="waterfall-container" style="height: 700px">
    <div
      v-for="(item, index) in list"
      :key="getKey(item, index)"
      :style="{
        backgroundColor,
      }"
      class="waterfall-item"
    >
      <div ref="imageRef" class="waterfall-card" @click="showBigPicture(index)">
        <LazyImg :url="getRenderURL(item)" />
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.waterfall-container {
  position: relative;
  display: grid;

  /* 自动创建多列，每列最小宽度200px，最大宽度为1fr（占据可用空间的一份） */
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));

  /* 自动行高，最小为0，根据内容自适应 */
  grid-auto-rows: minmax(0, auto);
  grid-gap: 10px;
  padding-bottom: 10px;
  overflow: auto;

  .waterfall-item {
    position: absolute;
  }

  .waterfall-card {
    cursor: pointer;
  }
}
</style>
