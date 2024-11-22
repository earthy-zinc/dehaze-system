<script lang="ts" setup>
import { useSettingsStore } from "@/store";
import Magnifier from "@/components/Magnifier/index.vue";
import { MagnifierInfo } from "../types";
import { useWindowSize } from "@vueuse/core";

defineOptions({
  name: "SubmitButton",
});

const props = defineProps({
  disableMore: {
    type: Boolean,
    default: false,
  },
  magnifier: {
    type: Object as () => MagnifierInfo,
    default: () => ({
      imgUrls: [],
      radius: 100,
      originScale: 1,
      point: {
        x: 0,
        y: 0,
      },
    }),
  },
});

const emit = defineEmits([
  "onGenerate",
  "onMagnifierChange",
  "onBrightnessChange",
  "onContrastChange",
]);

const { disableMore, magnifier } = toRefs(props);

const { themeColor } = useSettingsStore();

const { width } = useWindowSize();

const labelPosition = ref<"top" | "left" | "right">("left");
watch(
  () => width.value,
  (newValue) => {
    if (newValue >= 992) {
      labelPosition.value = "left";
    } else {
      labelPosition.value = "top";
    }
  }
);

const showMagnifier = ref(false);
const showContrast = ref(false);
const showBrightness = ref(false);
const brightness = ref(0);
const contrast = ref(0);
const magnifierShape = ref("square");
const magnifierScale = ref(8);
const magnifierLabels = ref([
  { text: "原图", color: "white", backgroundColor: "black" },
  { text: "对比图", color: "white", backgroundColor: themeColor },
]);

function handleMagnifierChange() {
  showMagnifier.value = !showMagnifier.value;
  emit("onMagnifierChange", showMagnifier.value);
}
</script>

<template>
  <div class="flex justify-evenly m-4">
    <el-button
      id="algorithm-generate-button"
      type="primary"
      @click="emit('onGenerate')"
    >
      立即生成
    </el-button>
    <el-dropdown :disabled="disableMore">
      <el-button :disabled="disableMore">更多功能</el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item @click="handleMagnifierChange">
            {{ showMagnifier ? "关闭" : "开启" }}放大镜
          </el-dropdown-item>
          <el-dropdown-item @click="showBrightness = !showBrightness">
            {{ showBrightness ? "关闭" : "开启" }}亮度调整
          </el-dropdown-item>
          <el-dropdown-item @click="showContrast = !showContrast">
            {{ showContrast ? "关闭" : "开启" }}对比度调整
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>

  <template v-if="showMagnifier && !disableMore">
    <div class="flex justify-between mb-3">
      <Magnifier
        v-for="(url, index) in magnifier.imgUrls"
        :key="url"
        :brightness="brightness"
        :contrast="contrast"
        :label="magnifierLabels[index]"
        :origin-scale="magnifier.originScale"
        :point="magnifier.point"
        :radius="magnifier.radius"
        :scale="magnifierScale"
        :shape="magnifierShape"
        :src="url"
      />
    </div>
  </template>

  <el-form :label-position="labelPosition" label-width="7.2vw">
    <template v-if="showMagnifier && !disableMore">
      <el-form-item class="more-operations" label="放大镜形状">
        <el-radio-group v-model="magnifierShape">
          <el-radio label="square">正方形</el-radio>
          <el-radio label="circle">圆形</el-radio>
        </el-radio-group>
      </el-form-item>

      <el-form-item class="more-operations" label="放大倍数">
        <el-slider v-model="magnifierScale" :max="20" :min="2" />
      </el-form-item>
    </template>

    <template v-if="showBrightness && !disableMore">
      <el-form-item class="more-operations" label="亮度">
        <el-slider
          v-model="brightness"
          :max="100"
          :min="-100"
          @change="(value) => emit('onBrightnessChange', value)"
        />
      </el-form-item>
    </template>

    <template v-if="showContrast && !disableMore">
      <el-form-item class="more-operations" label="对比度">
        <el-slider
          v-model="contrast"
          :max="100"
          :min="-100"
          @change="(value) => emit('onContrastChange', value)"
        />
      </el-form-item>
    </template>
  </el-form>
</template>

<style lang="scss">
.more-operations {
  margin-bottom: 5px;
}
</style>
