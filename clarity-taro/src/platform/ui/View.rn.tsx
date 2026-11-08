// 包入口导入：替换只发生在业务代码的深层 specifier 上（见 metro.config.js 的 resolveRequest），
// 这里不会被替换，因此不会自引用
import { View as TaroView } from "@tarojs/components-rn";
import { Text as RNText, StyleSheet } from "react-native";
import { LinearGradient } from "react-native-linear-gradient";
import type { ReactNode } from "react";
import { parseGradient } from "./gradient";

type Props = {
  style?: unknown;
  className?: string;
  children?: ReactNode;
  [key: string]: unknown;
};

/**
 * RN 端的 View：样式里出现 linear-gradient 时用 LinearGradient 渲染，其余情况透传给 Taro 的 View。
 *
 * 只在 taro build --type rn 时通过 config.rn.alias 生效，小程序与 H5 仍用原生 View。
 * 保留渐变是为了与 H5 视觉一致，不能用透明色代替。
 */
export default function View({ style, children, ...rest }: Props) {
  const flat = (StyleSheet.flatten(style as never) || {}) as Record<
    string,
    unknown
  >;
  const gradient = parseGradient(flat.backgroundImage as string | undefined);

  if (!gradient) {
    return (
      <TaroView style={style} {...rest}>
        {children}
      </TaroView>
    );
  }

  const { backgroundImage: _img, ...restStyle } = flat;
  // LinearGradient 是 RN 的 View，字符串子元素必须包在 Text 里
  const content =
    typeof children === "string" || typeof children === "number" ? (
      <RNText>{children}</RNText>
    ) : (
      children
    );

  return (
    <LinearGradient
      colors={gradient.colors}
      locations={gradient.locations}
      start={gradient.start}
      end={gradient.end}
      style={restStyle}
      {...(rest as object)}
    >
      {content}
    </LinearGradient>
  );
}
