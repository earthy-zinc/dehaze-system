<script lang="ts" setup>
import Magnifier from "@/components/Magnifier/index.vue";
import { PropType } from "vue";

const props = defineProps({
  imgUrls: {
    type: Array<string>,
    required: true,
  },
  point: {
    type: Object as PropType<{ x: number; y: number }>,
    required: true,
  },
  originScale: {
    type: Number,
    required: true,
  },
});

const emit = defineEmits([
  "update:contrast",
  "update:brightness",
  "toggle:magnifier",
  "toggle:takePhoto",
  "upload:image",
  "generate:image",
  "reset",
]);

const { width, height } = useWindowSize();

const magnifierInfo = reactive({
  enabled: false,
  shape: "square",
  radius: 110,
  scale: 8,
});

const contrast = ref(0);
const brightness = ref(0);

const showContrast = ref(false);
const showBrightness = ref(false);

function handleShowMagnifier() {
  magnifierInfo.enabled = !magnifierInfo.enabled;
  emit("toggle:magnifier", magnifierInfo.enabled);
}

function handleShowContrast() {
  showContrast.value = !showContrast.value;
}

function handleShowBrightness() {
  showBrightness.value = !showBrightness.value;
}

function handleTakePhoto() {
  emit("toggle:takePhoto");
}

function handleUploadChange(file: any) {}

function handleUploadExceed() {
  ElMessage.warning("最多只能上传一张图片");
}

onMounted(() => {
  magnifierInfo.radius = Math.floor((width.value * 0.35 - 90) / 4);
  watch(
    () => width.value,
    (newWidth) => {
      magnifierInfo.radius = Math.floor((newWidth * 0.35 - 90) / 4);
    }
  );
});
</script>

<template>
  <div ref="cardRef" class="mr-3">
    <el-card
      style="width: 35vw; height: 100%; padding: 0 15px; overflow-y: auto"
    >
      <!--标题及描述 -->
      <h2 class="text-center m-2">图像去雾</h2>
      <el-text class="m-2"
        >通过对遭受雾气影响的图像进行相应处理，恢复图像原本的纹理结构和细节信息，进而提升图像的能见度
      </el-text>

      <!--上传按钮 -->
      <div class="flex justify-evenly m-4">
        <el-upload
          :auto-upload="false"
          :limit="1"
          :on-change="handleUploadChange"
          :on-exceed="handleUploadExceed"
          :show-file-list="false"
          accept="image/gif,image/png,image/jpg,image/jpeg"
          action="#"
        >
          <el-button type="primary">上传图片</el-button>
        </el-upload>
        <el-button @click="emit('reset')">清除结果</el-button>
      </div>

      <!--提示词选择框 -->
      <div class="input">
        <el-text style="padding-left: 2px">
          输入提示词： 随机更换提示词
        </el-text>
        <el-input
          :autosize="{ minRows: 4, maxRows: 5 }"
          :rows="2"
          class="mt-2"
          clearable
          maxlength="500"
          placeholder="请输入提示词"
          show-word-limit
          type="textarea"
        />
        <el-icon
          style="position: relative; top: -21px; left: 10px; cursor: pointer"
        >
          <Delete />
        </el-icon>
      </div>

      <!--生成按钮及更多功能按钮 -->
      <div class="flex justify-evenly m-4">
        <el-button type="primary" @click="emit('generate:image')"
          >立即生成
        </el-button>
        <el-dropdown>
          <el-button type="primary">
            更多选项
            <el-icon>
              <arrow-down />
            </el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item @click="handleShowMagnifier">
                {{ magnifierInfo.enabled ? "关闭放大镜" : "开启放大镜" }}
              </el-dropdown-item>
              <el-dropdown-item @click="handleShowContrast">
                {{ showContrast ? "关闭对比度调整" : "开启对比度调整" }}
              </el-dropdown-item>
              <el-dropdown-item @click="handleShowBrightness">
                {{ showBrightness ? "关闭亮度调整" : "开启亮度调整" }}
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>

      <template v-if="magnifierInfo.enabled">
        <Magnifier
          v-for="imgUrl in props.imgUrls"
          :key="imgUrl"
          :origin-scale="originScale"
          :point="props.point"
          :radius="magnifierInfo.radius"
          :scale="magnifierInfo.scale"
          :shape="magnifierInfo.shape"
          :src="imgUrl"
        />
        <div class="flex flex-items-center w-90">
          <span class="w-35">放大镜形状</span>
          <el-radio-group v-model="magnifierInfo.shape">
            <el-radio-button label="square">正方形</el-radio-button>
            <el-radio-button label="circle">圆形</el-radio-button>
          </el-radio-group>
        </div>

        <div class="flex flex-items-center w-90 mt-3">
          <span class="w-35">放大倍数</span>
          <el-input-number
            v-model="magnifierInfo.scale"
            :max="20"
            :min="2"
            :step="0.1"
            :step-strictly="true"
          />
        </div>
      </template>

      <template v-if="showContrast">
        <div class="flex flex-items-center w-90 mt-3">
          <span class="w-35">对比度</span>
          <el-slider
            v-model="contrast"
            :max="100"
            :min="-100"
            @change="emit('update:contrast')"
          />
        </div>
      </template>

      <template v-if="showBrightness">
        <div class="flex flex-items-center w-90 mt-3">
          <span class="w-35">亮度</span>
          <el-slider
            v-model="brightness"
            :max="100"
            :min="-100"
            @change="emit('update:brightness')"
          />
        </div>
      </template>
    </el-card>
  </div>
</template>

<style lang="scss" scoped></style>
