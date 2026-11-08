import { Switch } from "react-native";

/**
 * Taroify Switch 的 RN 实现，由 metro.config.js 拦截 `@taroify/core/Switch` 生效。
 * 回调形状保持 Taroify 的 `onChange({ detail: { value } })`。
 */
interface Props {
  checked?: boolean;
  color?: string;
  size?: number | string;
  disabled?: boolean;
  className?: string;
  onChange?: (event: { detail: { value: boolean } }) => void;
  [key: string]: unknown;
}

export default function TaroifySwitch({
  checked,
  color = "#1989fa",
  disabled,
  onChange,
}: Props) {
  const size = 28;

  return (
    <Switch
      value={Boolean(checked)}
      onValueChange={(value) => onChange?.({ detail: { value } })}
      disabled={disabled}
      trackColor={{ false: "#dcdee0", true: color }}
      thumbColor="#fff"
      style={{
        transform: [{ scaleX: size / 31 }, { scaleY: size / 31 }],
      }}
    />
  );
}
