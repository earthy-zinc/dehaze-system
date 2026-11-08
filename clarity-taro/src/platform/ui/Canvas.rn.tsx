import { View } from "@tarojs/components";
import type { ComponentProps } from "react";

/**
 * POC 占位：RN 端没有 Canvas 组件，filter 页面的滤镜预览需要另走 RN 画布方案。
 * 这里只保证编译链路能解析，运行时不渲染任何内容。
 *
 * 只接样式相关属性：调用方传入的 type / id / canvasId 是 web 与小程序专有属性，
 * 整包透传给 RN 原生组件会产生未知属性告警。
 */
export default function Canvas({
  style,
  className,
}: Pick<ComponentProps<typeof View>, "style" | "className">) {
  return <View style={style} className={className} />;
}
