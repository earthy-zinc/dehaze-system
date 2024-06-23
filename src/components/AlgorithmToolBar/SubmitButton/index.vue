<script lang="ts" setup>
import { useSettingsStore } from "@/store";
import Magnifier from "@/components/Magnifier/index.vue";
import { MagnifierInfo } from "../types";

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
  emit("onMagnifierChange");
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
            {{ showMagnifier ? "开启" : "关闭" }}放大镜
          </el-dropdown-item>
          <el-dropdown-item @click="showBrightness = !showBrightness">
            {{ showBrightness ? "开启" : "关闭" }}亮度调整
          </el-dropdown-item>
          <el-dropdown-item @click="showContrast = !showContrast">
            {{ showBrightness ? "开启" : "关闭" }}对比度调整
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>

  <template v-if="showMagnifier">
    <div class="flex justify-between mb-3">
      <Magnifier
        v-for="(url, index) in magnifier.imgUrls"
        :src="url"
        :key="url"
        :label="magnifierLabels[index]"
        :brightness="brightness"
        :contrast="contrast"
        :shape="magnifierShape"
        :scale="magnifierScale"
        :point="magnifier.point"
        :radius="magnifier.radius"
        :origin-scale="magnifier.originScale"
      />
    </div>
  </template>

  <el-form label-width="7.2vw" label-position="left">
    <template v-if="showMagnifier">
      <el-form-item label="放大镜形状" class="more-operations">
        <el-radio-group v-model="magnifierShape">
          <el-radio label="square">正方形</el-radio>
          <el-radio label="circle">圆形</el-radio>
        </el-radio-group>
      </el-form-item>

      <el-form-item label="放大倍数" class="more-operations">
        <el-slider v-model="magnifierScale" :min="2" :max="20" />
      </el-form-item>
    </template>

    <template v-if="showBrightness">
      <el-form-item label="亮度" class="more-operations">
        <el-slider
          v-model="brightness"
          :min="-100"
          :max="100"
          @change="(value: number) => emit('onBrightnessChange', value)"
        />
      </el-form-item>
    </template>

    <template v-if="showContrast">
      <el-form-item label="对比度" class="more-operations">
        <el-slider
          v-model="contrast"
          :min="-100"
          :max="100"
          @change="(value: number) => emit('onContrastChange', value)"
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
