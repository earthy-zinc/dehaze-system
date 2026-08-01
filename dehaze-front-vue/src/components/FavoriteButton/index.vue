<template>
  <el-button
    :type="isFavorite ? 'primary' : 'default'"
    :icon="isFavorite ? StarFilled : Star"
    :loading="loading"
    :size="size === 'small' ? 'small' : size === 'large' ? 'large' : 'default'"
    @click="handleToggle"
  >
    <span v-if="showText">{{ isFavorite ? "已收藏" : "收藏" }}</span>
  </el-button>
</template>

<script lang="ts" setup>
import { ref, watch } from "vue";
import { FavoriteAPI, FavoriteQuery, FavoriteTargetType } from "dehaze-sdk-js";
import { Star, StarFilled } from "@element-plus/icons-vue";

defineOptions({ name: "FavoriteButton" });

const props = defineProps<{
  targetType: FavoriteTargetType;
  targetId: number;
  size?: "small" | "default" | "large";
  showText?: boolean;
}>();

const emit = defineEmits<{
  (e: "update", isFavorite: boolean): void;
}>();

const loading = ref(false);
const isFavorite = ref(false);
// Map of targetId -> favorite record id (needed for deleteByIds)
const favoriteIdMap = ref<Record<number, number>>({});

/** Fetch the favorite record id for the current target by querying the page */
async function resolveFavoriteId() {
  try {
    const query: FavoriteQuery = {
      pageNum: 1,
      pageSize: 10,
      targetType: props.targetType,
      keywords: undefined,
    };
    const data = await FavoriteAPI.getPage(query);
    const found = data.list.find((item) => item.targetId === props.targetId);
    if (found) {
      favoriteIdMap.value[props.targetId] = found.id;
    }
  } catch {
    // Ignore
  }
}

/** Load initial favorite status */
async function loadStatus() {
  loading.value = true;
  try {
    const status = await FavoriteAPI.getStatus(
      props.targetType,
      props.targetId
    );
    isFavorite.value = status.favorited;
    if (status.favorited) {
      // Need to resolve the record id via page query
      await resolveFavoriteId();
    } else {
      delete favoriteIdMap.value[props.targetId];
    }
  } catch {
    isFavorite.value = false;
    delete favoriteIdMap.value[props.targetId];
  } finally {
    loading.value = false;
  }
}

/** Delete by resolving the id if needed */
async function deleteFavorite() {
  const cachedId = favoriteIdMap.value[props.targetId];
  if (cachedId !== undefined) {
    await FavoriteAPI.deleteByIds([cachedId]);
    delete favoriteIdMap.value[props.targetId];
  } else {
    // Fallback: query again to find the id
    await resolveFavoriteId();
    const id = favoriteIdMap.value[props.targetId];
    if (id !== undefined) {
      await FavoriteAPI.deleteByIds([id]);
      delete favoriteIdMap.value[props.targetId];
    } else {
      ElMessage.warning("无法定位收藏记录，请稍后重试");
      return false;
    }
  }
  isFavorite.value = false;
  emit("update", false);
  ElMessage.success("已取消收藏");
  return true;
}

async function handleToggle() {
  if (loading.value) return;
  loading.value = true;

  try {
    if (isFavorite.value) {
      const success = await deleteFavorite();
      if (!success) return;
    } else {
      await FavoriteAPI.add({
        targetType: props.targetType,
        targetId: props.targetId,
      });
      isFavorite.value = true;
      favoriteIdMap.value[props.targetId] = -1; // placeholder, will resolve on next status check
      emit("update", true);
      ElMessage.success("收藏成功");
      // Resolve the actual id for future deletes
      await resolveFavoriteId();
    }
  } catch {
    ElMessage.warning("操作失败，请稍后重试");
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  loadStatus();
});

watch(
  () => [props.targetType, props.targetId],
  () => {
    loadStatus();
  }
);
</script>

<style scoped>
:deep(.el-button__icon) {
  font-size: inherit;
}
</style>
