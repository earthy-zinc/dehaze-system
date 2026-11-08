// 包入口导入：替换只发生在业务代码的深层 specifier 上（见 metro.config.js 的 resolveRequest），
// 这里不会被替换，因此不会自引用
import { Button as TaroButton } from "@tarojs/components-rn";
import { Pressable, StyleSheet, Text as RNText } from "react-native";
import { Children } from "react";
import { LinearGradient } from "react-native-linear-gradient";
import type { ReactNode } from "react";
import { parseGradient } from "./gradient";
import { isTaroifyButton, type ButtonProps } from "./button-shared";
import Loading from "./Loading";

/** Taroify 按钮的配色与尺寸（设计稿 750 下的值除以 2） */
const TAROIFY_COLORS: Record<
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

const TAROIFY_SIZES: Record<
  string,
  { height: number; fontSize: number; paddingHorizontal: number }
> = {
  mini: { height: 24, fontSize: 10, paddingHorizontal: 4 },
  small: { height: 32, fontSize: 12, paddingHorizontal: 8 },
  medium: { height: 44, fontSize: 14, paddingHorizontal: 16 },
  large: { height: 50, fontSize: 16, paddingHorizontal: 16 },
};

type Props = {
  style?: unknown;
  className?: string;
  children?: ReactNode;
  onClick?: (...args: unknown[]) => void;
  disabled?: boolean;
  [key: string]: unknown;
};

// CSS 的文本属性会向下继承，RN 不会，必须显式搬到内部 Text 上
const TEXT_KEYS = [
  "color",
  "fontSize",
  "fontWeight",
  "fontStyle",
  "fontFamily",
  "lineHeight",
  "letterSpacing",
  "textAlign",
  "textTransform",
  "textShadowColor",
  "textShadowOffset",
  "textShadowRadius",
] as const;

/**
 * RN 端的 Button：带渐变的按钮用 Pressable + LinearGradient 还原，其余透传给 Taro 的 Button。
 *
 * 渐变把背景画在绝对定位的图层上，容器自身负责 padding 与圆角裁剪，
 * 这样按钮的文字布局与不带渐变时完全一致。
 */
export default function Button({ style, children, onClick, ...rest }: Props) {
  const props = { style, onClick, ...rest } as ButtonProps;

  if (isTaroifyButton(props)) {
    const palette =
      TAROIFY_COLORS[rest.color as string] ?? TAROIFY_COLORS.default;
    const sizing = TAROIFY_SIZES[rest.size as string] ?? TAROIFY_SIZES.medium;
    const outlined = rest.variant === "outlined";

    return (
      <Pressable
        onPress={
          onClick
            ? (event) =>
                (onClick as (e: unknown) => void)({
                  ...(event as object),
                  stopPropagation: () => {},
                })
            : undefined
        }
        disabled={Boolean(rest.disabled)}
        style={[
          styles.taroifyButton,
          {
            height: sizing.height,
            paddingHorizontal: sizing.paddingHorizontal,
            borderRadius: 4,
          },
          rest.block || rest.size === "large"
            ? styles.taroifyBlock
            : styles.taroifyInline,
          outlined
            ? {
                backgroundColor: "transparent",
                borderWidth: 1,
                borderColor: palette.background,
              }
            : {
                backgroundColor: palette.background,
                borderWidth: 1,
                borderColor: palette.border,
              },
        ]}
      >
        {rest.loading ? <Loading size={sizing.fontSize} /> : null}
        {children ? (
          <RNText
            style={{
              color: outlined ? palette.background : palette.color,
              fontSize: sizing.fontSize,
              lineHeight: sizing.fontSize * 1.2,
              marginLeft: rest.loading ? 4 : 0,
            }}
          >
            {children as ReactNode}
          </RNText>
        ) : null}
      </Pressable>
    );
  }

  const flat = (StyleSheet.flatten(style as never) || {}) as Record<
    string,
    unknown
  >;
  const gradient = parseGradient(flat.backgroundImage as string | undefined);

  if (!gradient) {
    // 非渐变按钮也要补宽度：小程序 / H5 的 button 默认块级撑满，RN 的是内容宽
    const plain = { ...flat };
    if (plain.width === undefined) plain.width = "100%";
    // 文本属性同样要搬进内部 Text，否则 color / fontSize 这类继承属性会丢
    const plainText: Record<string, unknown> = {};
    for (const key of TEXT_KEYS) {
      if (plain[key] !== undefined) {
        plainText[key] = plain[key];
        delete plain[key];
      }
    }
    return (
      <TaroButton style={plain} onClick={onClick} {...rest}>
        {Children.map(children, (child) =>
          typeof child === "string" || typeof child === "number" ? (
            <RNText style={plainText as never}>{child}</RNText>
          ) : (
            child
          )
        )}
      </TaroButton>
    );
  }

  const { backgroundImage: _img, ...container } = flat;
  // 小程序 / H5 的 button 默认是块级元素、宽度撑满父容器；RN 的 Pressable 是内容宽。
  // 原样式没有显式宽度时补 100%，与 H5 的视觉结果对齐。
  if (container.width === undefined) {
    container.width = "100%";
  }
  const textStyle: Record<string, unknown> = {};
  for (const key of TEXT_KEYS) {
    if (container[key] !== undefined) {
      textStyle[key] = container[key];
      delete container[key];
    }
  }

  return (
    <Pressable
      onPress={
        onClick
          ? (event) =>
              (onClick as (e: unknown) => void)({
                ...(event as object),
                stopPropagation: () => {},
              })
          : undefined
      }
      style={container}
      {...(rest as object)}
    >
      <LinearGradient
        colors={gradient.colors}
        locations={gradient.locations}
        start={gradient.start}
        end={gradient.end}
        style={[
          StyleSheet.absoluteFill,
          { borderRadius: (container.borderRadius as number) || 0 },
        ]}
      />
      {Children.map(children, (child) =>
        typeof child === "string" || typeof child === "number" ? (
          <RNText style={textStyle as never}>{child}</RNText>
        ) : (
          child
        )
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  taroifyButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  taroifyInline: {
    alignSelf: "flex-start",
  },
  taroifyBlock: {
    alignSelf: "stretch",
  },
});
