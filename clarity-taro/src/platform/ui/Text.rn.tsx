// 包入口导入：替换只发生在业务代码的深层 specifier 上（见 metro.config.js 的 resolveRequest），
// 这里不会被替换，因此不会自引用
import { Text as TaroText } from "@tarojs/components-rn";
import { Text as RNText, StyleSheet, View } from "react-native";
// 默认导出叫 Svg，与包内的具名导出同名，别名避免 import/no-named-as-default
import SvgRoot, {
  Defs,
  LinearGradient as SvgLinearGradient,
  Stop,
  Text as SvgText,
  TSpan,
} from "react-native-svg";
import { useId, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { parseGradient, type ParsedGradient } from "./gradient";

type Props = {
  style?: unknown;
  className?: string;
  children?: ReactNode;
  [key: string]: unknown;
};

/** 一行文字的度量，单位与 RN 布局一致（dp） */
type TextLine = {
  text: string;
  /** RN 给出的基线（相对文字盒顶部），仅在拿不到字体的 ascent/descent 时兜底 */
  baseline: number;
  ascender?: number;
  descender?: number;
};

/**
 * RN 的 onTextLayout 行数据。Android 侧还会给 baseline / ascender / descender，
 * 但 TextLayoutEventData 的类型声明里没有 baseline。
 */
type RawLine = {
  text: string;
  y?: number;
  ascender?: number;
  descender?: number;
  baseline?: number;
};

/**
 * RN 端的 Text：带 background-clip: text 的渐变文字用 react-native-svg 的
 * <Text fill="url(#…)"> 绘制，其余情况透传给 Taro 的 Text。
 *
 * 早先的实现用 MaskedView + LinearGradient 做遮罩，Android 上文字右下方会多出一份
 * 半透明、错位的重影（实测：与渐变内容完全一致的第二份副本，位移约 1dp 量级）：
 * MaskedView 的 dispatchDraw 先画子节点、再在绘制过程中 setLayerType 切软件层，
 * 并用 View.layout() 强制重排遮罩后重新抓取遮罩位图，导致「遮罩 + 渐变子元素」
 * 被栅格化两次且两次的坐标基准不一致。子元素是纯色时看不出差异，多色 shader 才露馅。
 * 顺带它还把遮罩位图压扁了（实测字形高度只有正确的 70%）。
 *
 * SVG 让文字本身带渐变填充，不需要遮罩，也就不存在二次合成和位图缩放。
 */
export default function Text({ style, children, ...rest }: Props) {
  const flat = (StyleSheet.flatten(style as never) || {}) as Record<
    string,
    unknown
  >;
  const gradient = parseGradient(flat.backgroundImage as string | undefined);
  const isClipText = String(flat.backgroundClip || "").includes("text");

  if (!gradient || !isClipText) {
    return (
      <TaroText style={style} {...rest}>
        {children}
      </TaroText>
    );
  }

  return <GradientText gradient={gradient} flat={flat} content={children} />;
}

type GradientTextProps = {
  gradient: ParsedGradient;
  flat: Record<string, unknown>;
  content: ReactNode;
};

function GradientText({ gradient, flat, content }: GradientTextProps) {
  // 同一个页面上可能有多处渐变文字，渐变 id 必须唯一
  const gradientId = `text-grad-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const [width, setWidth] = useState(0);
  const [lines, setLines] = useState<TextLine[]>([]);

  const { backgroundImage: _img, backgroundClip: _clip, ...textStyle } = flat;

  const fontSize =
    typeof textStyle.fontSize === "number" ? textStyle.fontSize : 16;
  const lineHeight =
    typeof textStyle.lineHeight === "number"
      ? textStyle.lineHeight
      : fontSize * 1.2;

  // 换行位置由 RN 自己算：SVG 的 text 不支持自动换行，只能把 RN 量出来的每一行
  // 画成一条 TSpan。这样折行结果与不渐变时完全一致。
  // 水平方向不能用 onTextLayout 的 x —— Android 的 Layout.getLineLeft 对居中行
  // 恒返回 0，所以改用 textAlign 推导 textAnchor 来对齐。
  const renderLines = useMemo<TextLine[]>(() => {
    if (lines.length > 0) return lines;
    if (width <= 0) return [];
    // 兜底：onTextLayout 还没回来时至少把单行画对
    return [
      {
        text: typeof content === "string" ? content : "",
        baseline: fontSize * 0.85,
        ascender: fontSize * 0.85,
        descender: fontSize * 0.15,
      },
    ];
  }, [lines, width, content, fontSize]);

  const handleTextLayout = (event: { nativeEvent: { lines: RawLine[] } }) => {
    const next = (event.nativeEvent.lines || []).map((line, index) => ({
      text: line.text,
      baseline:
        typeof line.baseline === "number"
          ? line.baseline
          : (typeof line.y === "number" ? line.y : index * lineHeight) +
            (typeof line.ascender === "number"
              ? line.ascender
              : fontSize * 0.8),
      ascender: line.ascender,
      descender: line.descender,
    }));
    setLines((prev) =>
      prev.length === next.length &&
      prev.every(
        (line, index) =>
          line.text === next[index].text &&
          line.baseline === next[index].baseline
      )
        ? prev
        : next
    );
  };

  const handleLayout = (event: {
    nativeEvent: { layout: { width: number } };
  }) => {
    const { width: nextWidth } = event.nativeEvent.layout;
    setWidth((prev) => (prev === nextWidth ? prev : nextWidth));
  };

  const lineCount = Math.max(1, renderLines.length);
  // Taro 不会把 unitless line-height 透传给 RN（实测 style.lineHeight 是空的），
  // RN 的默认行高约 1.35em（含字体内边距），比 CSS 的 1.2em 高，会把下面的内容整体往下挤。
  // 这里按 CSS 语义把文字盒高度定为「行数 × 行高」，与 H5 的排版节奏对齐。
  const boxHeight = width > 0 ? lineHeight * lineCount : 0;

  // CSS 的行盒里基线位置 = 半行距 + ascent；RN 给的是 ascent（行盒顶部无额外 padding）。
  // 把基线换算到 CSS 的行盒后，行内垂直居中与多行行距才和 H5 一致。
  const baselineOf = (line: TextLine, index: number) => {
    const { ascender, descender } = line;
    if (
      typeof ascender === "number" &&
      typeof descender === "number" &&
      ascender + descender > 0
    ) {
      return (
        index * lineHeight + (lineHeight - ascender - descender) / 2 + ascender
      );
    }
    return line.baseline;
  };

  const fontWeight = textStyle.fontWeight as string | number | undefined;
  const fontFamily = textStyle.fontFamily as string | undefined;
  const fontStyle = textStyle.fontStyle as "normal" | "italic" | undefined;
  // SVG 的 letterSpacing 只认数值，em 单位在这里换算成 dp
  const letterSpacing = toSpacing(textStyle.letterSpacing, fontSize);

  // SVG 的 text 没有 textAlign，用 textAnchor + 锚点坐标等价还原
  const textAlign = String(textStyle.textAlign || "left");
  const textAnchor =
    textAlign === "center"
      ? "middle"
      : /end|right/.test(textAlign)
        ? "end"
        : "start";
  const anchorX =
    textAnchor === "middle" ? width / 2 : textAnchor === "end" ? width : 0;

  return (
    <View style={boxHeight > 0 ? { height: boxHeight } : undefined}>
      {/* 只负责排版与量尺寸：真正的字形由上层 Svg 绘制，避免两份文字叠在一起 */}
      <RNText
        style={[textStyle, styles.measure]}
        onLayout={handleLayout}
        onTextLayout={handleTextLayout}
      >
        {content}
      </RNText>
      {width > 0 && boxHeight > 0 && (
        <SvgRoot
          style={styles.svg}
          width={width}
          height={boxHeight}
          pointerEvents="none"
        >
          <Defs>
            {/* userSpaceOnUse：渐变范围按文字盒子的尺寸算，与 CSS 的
                linear-gradient 覆盖元素盒子一致，不受字形包围盒影响 */}
            <SvgLinearGradient
              id={gradientId}
              gradientUnits="userSpaceOnUse"
              x1={gradient.start.x * width}
              y1={gradient.start.y * boxHeight}
              x2={gradient.end.x * width}
              y2={gradient.end.y * boxHeight}
            >
              {gradient.colors.map((color, index) => (
                <Stop
                  key={color + index}
                  offset={gradient.locations[index]}
                  stopColor={color}
                />
              ))}
            </SvgLinearGradient>
          </Defs>
          <SvgText
            fill={`url(#${gradientId})`}
            textAnchor={textAnchor}
            fontSize={fontSize}
            fontWeight={fontWeight}
            fontFamily={fontFamily}
            fontStyle={fontStyle}
            letterSpacing={letterSpacing}
          >
            {renderLines.map((line, index) => (
              <TSpan key={index} x={anchorX} y={baselineOf(line, index)}>
                {line.text}
              </TSpan>
            ))}
          </SvgText>
        </SvgRoot>
      )}
    </View>
  );
}

/** 字间距归一成数值（dp）：SVG 只接受数字，em 需要按字号换算 */
function toSpacing(value: unknown, fontSize: number): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value !== "string") return undefined;
  const matched = /^(-?[\d.]+)(em)?$/.exec(value.trim());
  if (!matched) return undefined;
  const num = parseFloat(matched[1]);
  return matched[2] ? num * fontSize : num;
}

const styles = StyleSheet.create({
  // color: transparent：量尺寸用的文字本身不能可见，否则会和 SVG 的字形重影
  measure: { color: "transparent" },
  // 叠在量尺寸文字的位置上。尺寸由 width / height 显式给出，
  // 不能用 absoluteFill —— 那会被撑成父容器的尺寸（含外边距），SVG 随之被拉伸
  svg: { position: "absolute", left: 0, top: 0 },
});
