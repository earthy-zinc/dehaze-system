import { StyleSheet, TextInput } from "react-native";

/**
 * Taroify Input 的 RN 实现，由 metro.config.js 拦截 `@taroify/core/Input` 生效。
 * 回调形状保持 Taroify 的 `onInput({ detail: { value } })`，业务代码不用改。
 */
interface Props {
  value?: string;
  placeholder?: string;
  maxlength?: number;
  disabled?: boolean;
  type?: string;
  className?: string;
  style?: Record<string, unknown>;
  onInput?: (event: { detail: { value: string } }) => void;
  onChange?: (event: { detail: { value: string } }) => void;
  [key: string]: unknown;
}

export default function TaroifyInput({
  value,
  placeholder,
  maxlength,
  disabled,
  type,
  style,
  onInput,
  onChange,
}: Props) {
  const emit = (text: string) => {
    const event = { detail: { value: text } };
    onInput?.(event);
    onChange?.(event);
  };

  return (
    <TextInput
      value={value}
      placeholder={placeholder}
      placeholderTextColor="#c8c9cc"
      maxLength={maxlength}
      editable={!disabled}
      keyboardType={type === "number" ? "numeric" : "default"}
      onChangeText={emit}
      style={[styles.input, style as never]}
    />
  );
}

const styles = StyleSheet.create({
  input: {
    fontSize: 14,
    lineHeight: 20,
    color: "#323233",
    paddingVertical: 8,
    paddingHorizontal: 0,
  },
});
