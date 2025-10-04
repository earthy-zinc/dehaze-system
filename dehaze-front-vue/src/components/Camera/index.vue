<script lang="ts" setup>
defineOptions({
  name: "Camera",
});

const videoRef = ref<HTMLVideoElement>();
const canvasRef = ref<HTMLCanvasElement>();
const image = ref<File>();

const videoInfo = reactive({
  selectedDevice: {} as MediaDeviceInfo,
  device: [] as MediaDeviceInfo[],
  // video 元素实际展示宽高
  width: 0,
  height: 0,
  // 摄像头实际输出的分辨率
  videoWidth: 0,
  videoHeight: 0,
});

const isTakenPhoto = computed(() => {
  return image.value !== undefined;
});

const emit = defineEmits(["onSave", "onCancel"]);

async function startCamera(deviceId: string | undefined = undefined) {
  if (!videoRef.value) return;
  deviceId = deviceId || videoInfo.device[0].deviceId;
  try {
    videoRef.value.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { deviceId },
    });
    videoRef.value.onloadedmetadata = () => {
      if (!videoRef.value) return;
      let videoRect = videoRef.value!.getBoundingClientRect();
      videoInfo.width = videoRect.width;
      videoInfo.height = videoRect.height;
      videoInfo.videoHeight = videoRef.value!.videoHeight;
      videoInfo.videoWidth = videoRef.value!.videoWidth;
    };
  } catch (e: any) {
    ElMessage.error("加载摄像头失败", e.message);
  }
}

function capturePhoto() {
  if (!canvasRef.value) return;
  const canvas = canvasRef.value;
  canvas.width = videoInfo.width;
  canvas.height = videoInfo.height;

  const ctx = canvasRef.value.getContext("2d")!;
  ctx.clearRect(0, 0, videoInfo.width, videoInfo.height);
  ctx.drawImage(videoRef.value!, 0, 0, videoInfo.width, videoInfo.height);

  canvas.toBlob((blob) => {
    if (blob) {
      // 转化为file对象
      image.value = new File([blob], "image.png", {
        type: blob.type,
      });
    }
  });
  // 显示照片
  canvas.style.display = "block";
}

function retakePhoto() {
  if (!canvasRef.value) return;
  image.value = undefined;
  canvasRef.value.style.display = "none";
}

function savePhoto() {
  if (!image.value) return;
  emit("onSave", image.value);
}

function cancelTake() {
  emit("onCancel");
  stopCamera();
}

function stopCamera() {
  const stream = videoRef.value?.srcObject as MediaStream;
  if (stream) {
    stream.getTracks()?.forEach((track) => track.stop());
  }
}

function handleSelectChange(device: MediaDeviceInfo) {
  startCamera(device.deviceId);
}

onUnmounted(() => {
  stopCamera();
});

onMounted(async () => {
  try {
    let deviceInfos = await navigator.mediaDevices.enumerateDevices();
    videoInfo.device = deviceInfos.filter(
      (device) => device.kind === "videoinput"
    );
    videoInfo.selectedDevice = videoInfo.device[0];
    await startCamera();
  } catch (err: any) {
    ElMessage.error("无法获取摄像头信息，请检查设备", err.message);
  }
});
</script>

<template>
  <div>
    <div class="video-wrapper">
      <video ref="videoRef" autoplay></video>
      <canvas ref="canvasRef"></canvas>
    </div>

    <div
      :style="{ paddingTop: `${videoInfo.height + 15}px` }"
      class="controller"
    >
      <span class="mr-2" style="line-height: 32px">切换摄像头</span>
      <el-select
        v-model="videoInfo.selectedDevice"
        :disabled="isTakenPhoto"
        class="mr-6"
        placeholder="选择摄像头"
        style="width: 200px"
        value-key="deviceId"
        @change="handleSelectChange"
      >
        <el-option
          v-for="device in videoInfo.device"
          :key="device.deviceId"
          :label="device.label"
          :value="device"
        />
      </el-select>
      <ElButton
        v-show="!isTakenPhoto"
        class="ml-4"
        type="primary"
        @click="capturePhoto"
        >拍照
      </ElButton>
      <ElButton v-show="isTakenPhoto" @click="retakePhoto">重拍</ElButton>
      <ElButton v-show="isTakenPhoto" type="primary" @click="savePhoto"
        >保存
      </ElButton>
      <ElButton @click="cancelTake">取消</ElButton>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.video-wrapper {
  position: relative;

  video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: auto;
  }

  canvas {
    position: absolute;
    top: 0;
    left: 0;
    display: none;
  }
}

.controller {
  position: relative;
  display: flex;
  justify-content: center;
}
</style>
