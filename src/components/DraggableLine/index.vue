<script lang="ts" setup>
import { useSettingsStore } from "@/store";
import { hexToRGBA } from "@/utils";

const emit = defineEmits(["update:offset"]);
const isDragging = ref(false);
// 线段离父容器最左侧的距离
const parentOffsetLeft = ref(0);
// 父容器离HTML文档最左侧的距离
const offsetLeft = ref(0);
const dragContainer = ref<HTMLElement>();

const { width, height } = useWindowSize();
const settingStore = useSettingsStore();

const props = defineProps({
  leftLabel: {
    type: String,
  },
  rightLabel: {
    type: String,
  },
});

/**
 * 绑定鼠标事件
 */
function drag(event: MouseEvent) {
  if (isDragging.value) {
    offsetLeft.value = dragContainer.value!.getBoundingClientRect().left;
    parentOffsetLeft.value = event.clientX - offsetLeft.value;
  }
}

watch(parentOffsetLeft, (newValue) => {
  emit("update:offset", newValue);
});

onMounted(() => {
  const { left } = dragContainer.value!.getBoundingClientRect();
  parentOffsetLeft.value = dragContainer.value!.offsetWidth;
  offsetLeft.value = left;
});
</script>

<template>
  <div
    ref="dragContainer"
    class="container"
    @mousedown="isDragging = true"
    @mousemove="drag"
    @mouseup="isDragging = false"
  >
    <div
      :style="{
        left: `${parentOffsetLeft}px`,
        backgroundColor: settingStore.themeColor,
      }"
      class="line"
    >
      <svg-icon
        :color="settingStore.themeColor"
        class="icon-location"
        icon-class="hollow-slide"
        size="2em"
      />
    </div>
    <div
      v-show="leftLabel"
      :style="{
        left: `${parentOffsetLeft - 80}px`,
        backgroundColor: 'rgba(162,162,162,0.5)',
        color: 'var(--el-border-color)',
      }"
      class="drag-label"
    >
      <span>{{ leftLabel }}</span>
    </div>
    <div
      v-show="rightLabel"
      :style="{
        left: `${parentOffsetLeft}px`,
        backgroundColor: hexToRGBA(settingStore.themeColor, 0.5),
        color: 'var(--el-border-color)',
      }"
      class="drag-label"
    >
      <span>{{ rightLabel }}</span>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.container {
  position: relative;
  height: 100%;
}

.drag-label {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 80px;
  height: 30px;
  line-height: 30px;
  text-align: center;
}

.line {
  position: absolute;
  top: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 3px;
}

.line .icon-location {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
}
</style>
