import {
  Pressable,
  StyleSheet,
  Text as RNText,
  type TextStyle,
} from "react-native";

import { ICON_FONT_FAMILY, ICON_UNICODE } from "./icon-map";
import type { IconProps } from "./icon-shared";

/**
 * RN 端的图标：把 Taroify 的 iconfont 码点直接渲染成字符。
 * 字体文件在 rn/android/app/src/main/assets/fonts/vant-icon.ttf，与 Taroify 用的是同一份，
 * 因此三端图形完全一致。
 */
export default function Icon({ name, size, color, style, onClick }: IconProps) {
  const glyph = ICON_UNICODE[name];
  if (!glyph) return null;

  const glyphStyle: TextStyle = {
    fontFamily: ICON_FONT_FAMILY,
    fontSize: size === undefined ? undefined : Number(size),
  };

  const merged = StyleSheet.flatten([
    glyphStyle,
    color ? { color } : null,
    style as TextStyle,
  ]);

  const text = <RNText style={merged}>{glyph}</RNText>;

  if (!onClick) return text;

  // 图标经常当按钮用（例如列表行的新增），没有 onClick 时不额外包一层，避免影响布局
  return (
    <Pressable onPress={onClick} hitSlop={8}>
      {text}
    </Pressable>
  );
}
