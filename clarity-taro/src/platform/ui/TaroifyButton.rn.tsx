import { Pressable, StyleSheet, Text } from "react-native";

import Loading from "./Loading.rn";

/**
 * Taroify Button 的 RN 实现，由 metro.config.js 拦截 `@taroify/core/Button` 生效。
 * 小程序与 H5 不拦截，仍用 Taroify 原组件，因此 web 端零风险。
 *
 * 样式取自 Taroify 的 button 样式表（设计稿 750 的值除以 2）。
 */
const COLORS: Record<
  string,
  { background: string; border: string; color: string }
> = {
  default: { background: "#fff", border: "#dcdee0", color: "#323233" },
  primary: { background: "#1989fa", border: "#1989fa", color: "#fff" },
  info: { background: "#2db7f5", border: "#2db7f5", color: "#fff" },
  success: { background: "#07c160", border: "#07c160", color: "#fff" },
  warning: { background: "#ff976a", border: "#ff976a", color: "#fff" },
  danger: { background: "#ee0a24", border: "#ee0a24", color: "#fff" },
};

const SIZES: Record<
  string,
  { height: number; fontSize: number; paddingHorizontal: number }
> = {
  mini: { height: 24, fontSize: 10, paddingHorizontal: 4 },
  small: { height: 32, fontSize: 12, paddingHorizontal: 8 },
  medium: { height: 44, fontSize: 14, paddingHorizontal: 16 },
  large: { height: 50, fontSize: 16, paddingHorizontal: 16 },
};

interface Props {
  color?: keyof typeof COLORS;
  size?: keyof typeof SIZES;
  block?: boolean;
  variant?: "outlined" | "contained";
  loading?: boolean;
  disabled?: boolean;
  onClick?: (event: { stopPropagation: () => void }) => void;
  children?: React.ReactNode;
  className?: string;
  style?: Record<string, unknown>;
  [key: string]: unknown;
}

export default function TaroifyButton({
  color = "default",
  size = "medium",
  block,
  variant,
  loading,
  disabled,
  onClick,
  children,
  style,
}: Props) {
  const palette = COLORS[color] ?? COLORS.default;
  const sizing = SIZES[size] ?? SIZES.medium;
  const outlined = variant === "outlined";

  return (
    <Pressable
      onPress={
        onClick ? () => onClick({ stopPropagation: () => {} }) : undefined
      }
      disabled={disabled || loading}
      style={[
        styles.base,
        {
          height: sizing.height,
          paddingHorizontal: sizing.paddingHorizontal,
        },
        block || size === "large" ? styles.block : styles.inline,
        outlined
          ? { backgroundColor: "transparent", borderColor: palette.background }
          : {
              backgroundColor: palette.background,
              borderColor: palette.border,
            },
        style as never,
      ]}
    >
      {loading ? <Loading size={sizing.fontSize} /> : null}
      {children ? (
        <Text
          style={{
            color: outlined ? palette.background : palette.color,
            fontSize: sizing.fontSize,
            lineHeight: sizing.fontSize * 1.2,
            marginLeft: loading ? 4 : 0,
          }}
        >
          {children as React.ReactNode}
        </Text>
      ) : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 4,
    borderWidth: 1,
  },
  inline: {
    alignSelf: "flex-start",
  },
  block: {
    alignSelf: "stretch",
  },
});
