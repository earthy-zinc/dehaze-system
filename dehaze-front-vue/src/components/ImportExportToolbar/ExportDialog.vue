<script lang="ts" setup>
import { ExportModule } from "dehaze-sdk-js";
import { useImportExport } from "@/composables/useImportExport";

defineOptions({
  name: "ExportDialog",
});

const props = defineProps<{
  modelValue: boolean;
  module: ExportModule;
  queryParams: Record<string, any>;
  fields?: { label: string; value: string }[];
  initialFormat?: "excel" | "csv";
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
  (e: "async-task-created", taskId: string): void;
}>();

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit("update:modelValue", val),
});

const moduleRef = computed(() => props.module);
const queryParamsRef = computed(() => props.queryParams);

const { exportLoading, exportData, downloadExportBlob } = useImportExport({
  module: moduleRef,
  queryParams: queryParamsRef,
});

const format = ref<"excel" | "csv">("excel");
const selectedFields = ref<string[]>([]);
const forceAsync = ref(false);
const asyncTaskId = ref<string | null>(null);

const hasFieldConfig = computed(
  () => !!props.fields && props.fields.length > 0
);

watch(
  () => props.modelValue,
  (val) => {
    if (val) {
      format.value = props.initialFormat ?? "excel";
    }
  },
  { immediate: true }
);

function resetState() {
  selectedFields.value = [];
  forceAsync.value = false;
  asyncTaskId.value = null;
}

function handleClose() {
  resetState();
  visible.value = false;
}

async function handleSubmit() {
  try {
    const result = await exportData(
      format.value,
      hasFieldConfig.value ? selectedFields.value : undefined,
      forceAsync.value
    );
    if (result.isAsync) {
      asyncTaskId.value = result.taskId!;
      ElMessage.success("数据量较大，已创建导出任务，可在任务列表查看进度");
      emit("async-task-created", result.taskId!);
      handleClose();
    } else if (result.blob) {
      downloadExportBlob(result.blob, format.value);
      ElMessage.success("导出成功");
      handleClose();
    }
  } catch (e: any) {
    ElMessage.error(e.message || "导出失败");
  }
}

const canSubmit = computed(() => !exportLoading.value);
</script>

<template>
  <el-dialog
    v-model="visible"
    title="导出数据"
    width="560px"
    :close-on-click-modal="false"
    append-to-body
    @close="handleClose"
  >
    <div v-if="!asyncTaskId" class="export-form">
      <el-form label-width="100px">
        <el-form-item label="文件格式">
          <el-radio-group v-model="format">
            <el-radio value="excel">Excel (.xlsx)</el-radio>
            <el-radio value="csv">CSV (.csv)</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item v-if="hasFieldConfig" label="导出字段">
          <div class="field-selector">
            <el-checkbox
              :model-value="
                selectedFields.length === (fields?.length ?? 0) &&
                selectedFields.length > 0
              "
              :indeterminate="
                selectedFields.length > 0 &&
                selectedFields.length < (fields?.length ?? 0)
              "
              @change="
                (val: string | number | boolean) =>
                  (selectedFields = val
                    ? (fields ?? []).map((f) => f.value)
                    : [])
              "
            >
              全选
            </el-checkbox>
            <el-divider class="field-divider" />
            <el-checkbox-group v-model="selectedFields" class="field-group">
              <el-checkbox
                v-for="field in fields"
                :key="field.value"
                :label="field.label"
                :value="field.value"
              />
            </el-checkbox-group>
            <div class="field-tip">不勾选则导出全部字段</div>
          </div>
        </el-form-item>

        <el-form-item label="异步导出">
          <el-switch v-model="forceAsync" />
          <div class="async-tip">
            开启后强制走异步任务，适用于大数据量导出（单次最多 10 万条）
          </div>
        </el-form-item>
      </el-form>
    </div>

    <el-result
      v-else
      icon="info"
      title="已创建导出任务"
      sub-title="数据量较大，已转为异步任务，可在任务列表查看进度"
    />

    <template #footer>
      <el-button @click="handleClose">
        {{ asyncTaskId ? "关闭" : "取消" }}
      </el-button>
      <el-button
        v-if="!asyncTaskId"
        type="primary"
        :loading="exportLoading"
        :disabled="!canSubmit"
        @click="handleSubmit"
      >
        确定导出
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.export-form {
  .field-selector {
    width: 100%;
  }

  .field-divider {
    margin: 8px 0;
  }

  .field-group {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
  }

  .field-tip,
  .async-tip {
    margin-top: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}
</style>
