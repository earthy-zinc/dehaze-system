import {
  Pressable,
  StyleSheet,
  Text,
  View,
  type TextStyle,
} from "react-native";

import type { TagProps } from "./tag-shared";

/** 与 Taroify 的 tag 样式表一致（设计稿 750 下的值除以 2 得到 dp） */
const COLORS: Record<string, string> = {
  default: "#969799",
  primary: "#1989fa",
  info: "#2db7f5",
  success: "#07c160",
  warning: "#ff976a",
  danger: "#ee0a24",
};

const SIZES: Record<
  string,
  TextStyle & { paddingHorizontal: number; borderRadius: number }
> = {
  small: {
    fontSize: 12,
    lineHeight: 16,
    paddingHorizontal: 4,
    borderRadius: 2,
    paddingVertical: 0,
  },
  medium: {
    fontSize: 12,
    lineHeight: 16,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 2,
  },
  large: {
    fontSize: 14,
    lineHeight: 18,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
};

export default function Tag({
  color = "default",
  size = "small",
  variant,
  children,
  onClick,
}: TagProps) {
  const background = COLORS[color] ?? COLORS.default;
  const sizing = SIZES[size] ?? SIZES.small;
  const outlined = variant === "outlined";
  const Wrapper = onClick ? Pressable : View;

  return (
    <Wrapper
      onPress={onClick}
      style={[
        styles.tag,
        sizing,
        outlined
          ? {
              backgroundColor: "transparent",
              borderWidth: 1,
              borderColor: background,
            }
          : { backgroundColor: background },
      ]}
    >
      <Text
        style={[
          styles.text,
          outlined ? { color: background } : null,
          { fontSize: sizing.fontSize, lineHeight: sizing.lineHeight },
        ]}
      >
        {children}
      </Text>
    </Wrapper>
  );
}

const styles = StyleSheet.create({
  tag: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
  },
  text: {
    color: "#fff",
  },
});
