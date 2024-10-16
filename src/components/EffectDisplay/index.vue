<script setup lang="ts">
import { onMounted } from "vue";
import OverlapImageShow from "@/components/OverlapImageShow/index.vue";
import { Point } from "@/components/AlgorithmToolBar/types";

defineOptions({
  name: "EffectDisplay",
});

const props = defineProps({
  urls: {
    type: Array as PropType<string[]>,
    required: true,
  },
});

const image1 = ref("");
const image2 = ref("");
const point = ref<Point>({
  x: 0,
  y: 0,
});

function handleMouseover(p: Point) {
  point.value.x = p.x;
  point.value.y = p.y;
}

onMounted(() => {
  image1.value = props.urls[0];
  image2.value = props.urls[1];
});
</script>

<template>
  <div class="effect-container">
    <h2>效果展示</h2>
    <p>去雾前后对比</p>
    <div class="display-wrap">
      <div class="display-container" v-for="index in 3" :key="index">
        <div class="overlap-wrap">
          <!-- 重叠展示 -->
          <OverlapImageShow
            class="overlap"
            :image1="image1"
            :image2="image2"
            :height="200"
            @on-origin-scale-change="(value) => (originScale = value)"
            @on-mouseover="handleMouseover"
          />
        </div>
        <div class="user-info-wrap">
          <img
            src="https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif?imageView2/1/w/80/h/80"
            alt="avatar"
          />
          <div class="right-wrap">
            <p>admin</p>
            <p>2024-07-08 · 10:20:11</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.effect-container {
  width: 100%;
  min-width: 300px;
  height: auto;
  border: 2px solid var(--el-color-primary);

  h2 + p,
  h2 {
    width: 100%;
    height: 30px;
    text-align: center;
  }

  .display-wrap {
    display: flex;
    justify-content: space-around;

    .display-container {
      width: 30%;
      height: auto;

      .user-info-wrap {
        display: flex;
        height: 80px;
        margin-top: 10px;

        img {
          width: 42px;
          height: 42px;
          vertical-align: middle;
          object-fit: inherit;
          border-radius: 50%;
        }

        .right-wrap {
          height: 42px;

          p {
            margin: 0 0 0 10px;

            &:nth-child(1) {
              font-weight: 700;
            }

            &:nth-child(2) {
              font-weight: 500;
              color: var(--el-color-info);
            }
          }
        }
      }
    }
  }
}

@media screen and (width <= 992px) {
  .display-wrap {
    flex-direction: column;

    .display-container {
      display: flex;
      flex-direction: column;
      justify-content: space-around;
      width: 100%;
      min-width: 300px;
      margin: 0 auto;
    }
  }
}
</style>
