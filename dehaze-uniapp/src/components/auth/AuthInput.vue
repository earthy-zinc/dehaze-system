<template>
  <view class="auth-input">
    <view class="input-wrapper" :class="{ focused: focused, error: !!error }">
      <u-icon
        v-if="icon"
        :name="icon"
        size="20"
        color="#9ca3af"
        class="input-icon"
      />
      <input
        :value="modelValue"
        :type="inputType"
        :password="password && !showPassword"
        :placeholder="placeholder"
        placeholder-class="placeholder"
        :maxlength="maxlength"
        @input="onInput"
        @focus="focused = true"
        @blur="focused = false"
      />
      <view
        v-if="password"
        class="eye-btn"
        @click="showPassword = !showPassword"
      >
        <u-icon
          :name="showPassword ? 'eye-fill' : 'eye-off'"
          size="20"
          color="#9ca3af"
        />
      </view>
    </view>
    <text v-if="error" class="error-message">
      <u-icon name="info-circle" size="12" color="#ef4444" />
      <text>{{ error }}</text>
    </text>
  </view>
</template>

<script lang="ts" setup>
import { ref, computed } from "vue";

interface Props {
  /** v-model 绑定值 */
  modelValue: string;
  /** 输入框左侧图标（uview-plus 图标名） */
  icon?: string;
  /** 占位文案 */
  placeholder?: string;
  /** 是否为密码框（显示眼睛切换） */
  password?: boolean;
  /** 错误提示（非空时输入框标红并展示） */
  error?: string;
  /** 最大长度 */
  maxlength?: number;
  /** 输入类型（非密码框时生效） */
  type?: string;
}

const props = withDefaults(defineProps<Props>(), {
  icon: "",
  placeholder: "",
  password: false,
  error: "",
  maxlength: 140,
  type: "text",
});

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
}>();

const focused = ref(false);
const showPassword = ref(false);

/** 实际输入类型：密码框始终 text，由 password 属性控制掩码 */
const inputType = computed(() => (props.password ? "text" : props.type));

function onInput(e: any) {
  emit("update:modelValue", e.detail.value);
}
</script>

<style lang="scss" scoped>
.auth-input {
  margin-bottom: 24rpx;
}

.input-wrapper {
  display: flex;
  align-items: center;
  padding: 0 24rpx;
  height: 96rpx;
  background: $color-bg-primary;
  border: 2rpx solid $color-border;
  border-radius: $radius-sm;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);

  .input-icon {
    margin-right: 16rpx;
    flex-shrink: 0;
    transition: color 0.2s ease;
  }

  input {
    flex: 1;
    font-size: 28rpx;
    color: $color-text-primary;
    background: transparent;
    border: none;
    outline: none;
    height: 100%;
  }

  .placeholder {
    color: $color-text-placeholder;
    font-size: 28rpx;
  }

  .eye-btn {
    width: 56rpx;
    height: 56rpx;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: 8rpx;
    flex-shrink: 0;

    &:active {
      opacity: 0.6;
    }
  }

  &.focused {
    background: $color-white;
    border-color: $color-primary;
    box-shadow: $shadow-input-focus;

    .input-icon {
      color: $color-primary;
    }
  }

  &.error {
    border-color: $color-danger;
    background: $color-danger-bg;
  }
}

.error-message {
  display: flex;
  align-items: center;
  gap: 6rpx;
  margin-top: 12rpx;
  font-size: 22rpx;
  color: $color-danger;
}
</style>
