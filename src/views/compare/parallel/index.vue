<script setup lang="ts">
const glassRef = ref<HTMLElement>();
const bigImgRef = ref<HTMLImageElement>();
const glassWrapperRef = ref<HTMLElement>();

function handleMouseOverImage(event: MouseEvent) {
  glassRef.value!.style.display = "block";
  bigImgRef.value!.style.display = "block";
}
function handleMouseOutImage(event: MouseEvent) {
  glassRef.value!.style.display = "none";
  bigImgRef.value!.style.display = "none";
}
function handleMouseMove(event: MouseEvent) {
  // 该操作让glassWrapper的左上角变成坐标原点, 因为glass是先相对于glassWrapper而移动的
  const x = event.pageX - glassWrapperRef.value!.offsetLeft;
  const y = event.pageY - glassWrapperRef.value!.offsetTop;
  // 让鼠标在glass的中间位置
  let width = x - glassRef.value!.offsetWidth / 2;
  let height = y - glassRef.value!.offsetHeight / 2;
  // 让glass不超出img内部
  if (width <= 0) {
    width = 0;
  } else if (
    width >=
    glassWrapperRef.value!.offsetWidth - glassRef.value!.offsetWidth
  ) {
    width = glassWrapperRef.value!.offsetWidth - glassRef.value!.offsetWidth;
  }
  if (height <= 0) {
    height = 0;
  } else if (
    height >=
    glassWrapperRef.value!.offsetHeight - glassRef.value!.offsetHeight
  ) {
    height = glassWrapperRef.value!.offsetHeight - glassRef.value!.offsetHeight;
  }

  // 改变放大镜的位置
  glassRef.value!.style.left = width + "px";
  glassRef.value!.style.top = height + "px";
}
</script>

<template>
  <div class="box" @mousemove="handleMouseMove">
    <div
      @mouseover="handleMouseOverImage"
      @mouseout="handleMouseOutImage"
      ref="glassWrapperRef"
      class="glassWrapper"
    >
      <img src="https://picsum.photos/id/1/800/800" class="img" />
      <div ref="glassRef" class="glass" id="glass"></div>
    </div>
    <div class="bigWrapper">
      <img
        ref="bigImgRef"
        src="https://picsum.photos/id/1/800/800"
        class="bigImg"
      />
    </div>
  </div>
</template>

<style scoped lang="scss">
.glassTitle {
  color: #89cff0;
  text-align: center;
}

.box {
  display: flex;
  align-items: center;
  justify-content: space-around;
  width: 80vw;
  min-width: 800px;
  height: 80vh;
  min-height: 600px;
  margin: 10px auto;
  line-height: 80vh;
  background-color: #f2f3f4;
  border-radius: 10px;
  box-shadow: 0 0 10px 1px #5d8aa8;
}

.glassWrapper {
  position: relative;
  line-height: 0;
}

.img {
  display: block;
  width: 250px;
  height: auto;
}

.glass {
  position: absolute;
  display: none;
  width: 80px;
  height: 80px;
  background: #89cff0;
  opacity: 0.5;
}

.bigWrapper {
  position: relative;
  width: 500px;
  height: 500px;
  background-color: #fff;
  border: 1px dashed #89cff0;
  border-radius: 10px;
  //overflow: hidden;
}

.bigImg {
  position: absolute;
  display: none;
  width: 2500px;
}
</style>
