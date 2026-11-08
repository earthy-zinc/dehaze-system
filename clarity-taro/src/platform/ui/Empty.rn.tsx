import {
  Image,
  StyleSheet,
  Text,
  View,
  type ImageStyle,
  type TextStyle,
  type ViewStyle,
} from "react-native";

import type {
  EmptyDescriptionProps,
  EmptyImageProps,
  EmptyProps,
} from "./empty-shared";

/** 与 Taroify 一致：预置图名映射到官方图床，其余按原地址加载 */
const PRESET_IMAGES = ["default", "error", "network", "search"] as const;

function resolveImageUrl(src?: string): string {
  if (!src) return "https://img.yzcdn.cn/vant/empty-image-default.png";
  if ((PRESET_IMAGES as readonly string[]).includes(src)) {
    return `https://img.yzcdn.cn/vant/empty-image-${src}.png`;
  }
  return src;
}

/** 设计稿 750 下的数值除以 2 得到 dp */
function EmptyLayout({ children }: EmptyProps) {
  return <View style={styles.container}>{children}</View>;
}

function EmptyImage({ src }: EmptyImageProps) {
  return (
    <Image
      source={{ uri: resolveImageUrl(src) }}
      style={styles.image as ImageStyle}
      resizeMode="contain"
    />
  );
}

function EmptyDescription({ children }: EmptyDescriptionProps) {
  return <Text style={styles.description as TextStyle}>{children}</Text>;
}

EmptyLayout.Image = EmptyImage;
EmptyLayout.Description = EmptyDescription;

export default EmptyLayout as React.FC<EmptyProps> & {
  Image: React.FC<EmptyImageProps>;
  Description: React.FC<EmptyDescriptionProps>;
};

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    paddingVertical: 32,
  } as ViewStyle,
  image: {
    width: 160,
    height: 160,
  } as ImageStyle,
  description: {
    color: "#969799",
    fontSize: 14,
    lineHeight: 20,
    marginTop: 16,
    paddingHorizontal: 60,
    textAlign: "center",
  } as TextStyle,
});
