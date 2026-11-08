import { useEffect, useRef } from "react";
import { Animated, Easing, StyleSheet, Text, View } from "react-native";

import type { LoadingProps } from "./loading-shared";

/** 设计稿 750 下的尺寸除以 2 得到 dp */
function toDp(size: LoadingProps["size"]): number {
  if (size === undefined) return 30;
  if (size === "small") return 10;
  if (size === "medium") return 20;
  if (size === "large") return 30;
  const value = Number(size);
  return Number.isFinite(value) ? value / 2 : 30;
}

const SPOKES = 12;

/**
 * 与 Taroify 的 spinner 同构：12 根线条按 30 度排布、透明度递减，整体匀速旋转。
 * 原版是 steps(12) 的跳变，这里用连续旋转，观感更平滑。
 */
export default function Loading({
  size,
  color = "#c8c9cc",
  children,
}: LoadingProps) {
  const dimension = toDp(size);
  const spin = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.timing(spin, {
        toValue: 1,
        duration: 800,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    );
    animation.start();
    return () => animation.stop();
  }, [spin]);

  const rotate = spin.interpolate({
    inputRange: [0, 1],
    outputRange: ["0deg", "360deg"],
  });

  const spinner = (
    <Animated.View
      style={[
        styles.container,
        { width: dimension, height: dimension, transform: [{ rotate }] },
      ]}
    >
      {Array.from({ length: SPOKES }, (_, index) => (
        <View
          key={index}
          style={[
            styles.spoke,
            {
              transform: [{ rotate: `${index * (360 / SPOKES)}deg` }],
              opacity: 1 - index * (1 / SPOKES),
            },
          ]}
        >
          <View
            style={[
              styles.bar,
              { backgroundColor: color, height: Math.max(dimension * 0.25, 2) },
            ]}
          />
        </View>
      ))}
    </Animated.View>
  );

  if (!children) return spinner;

  return (
    <View style={styles.withText}>
      {spinner}
      <Text style={[styles.label, { color }]}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
  },
  spoke: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
  },
  bar: {
    width: 2,
    borderRadius: 1,
  },
  withText: {
    flexDirection: "row",
    alignItems: "center",
  },
  label: {
    marginLeft: 6,
    fontSize: 12,
  },
});
