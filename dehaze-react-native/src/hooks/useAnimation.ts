import { useRef, useEffect, useMemo, useCallback } from 'react';
import { Animated, Easing, type ViewStyle } from 'react-native';

interface FadeSlideOptions {
  duration?: number;
  slideDistance?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
  autoStart?: boolean;
  delay?: number;
  easing?: (value: number) => number;
  scale?: {
    initial: number;
    final: number;
  };
}

/**
 * 组合动画 Hook：淡入 + 滑动 + (可选)缩放
 */
export function useFadeSlideAnimation(options: FadeSlideOptions = {}) {
  const {
    duration = 800,
    slideDistance = 30,
    direction = 'up',
    autoStart = true,
    delay = 0,
    easing = Easing.out(Easing.cubic),
    scale,
  } = options;
  
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(
    direction === 'down' || direction === 'right' ? -slideDistance : slideDistance
  )).current;
  const scaleAnim = useRef(new Animated.Value(scale?.initial ?? 1)).current;

  const start = useCallback(() => {
    const animations: Animated.CompositeAnimation[] = [
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration,
        easing,
        useNativeDriver: true,
        delay,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration,
        easing,
        useNativeDriver: true,
        delay,
      }),
    ];

    if (scale) {
      animations.push(
        Animated.timing(scaleAnim, {
          toValue: scale.final,
          duration,
          easing,
          useNativeDriver: true,
          delay,
        })
      );
    }

    Animated.parallel(animations).start();
  }, [fadeAnim, slideAnim, scaleAnim, duration, delay, easing, scale]);

  useEffect(() => {
    if (autoStart) {
      start();
    }
  }, [autoStart, start]);

  const animatedStyle = useMemo(() => {
    const transform: Array<{ translateX?: Animated.Value } | { translateY?: Animated.Value } | { scale?: Animated.Value }> = [];

    if (slideDistance > 0) {
      if (direction === 'left' || direction === 'right') {
        transform.push({ translateX: slideAnim });
      } else {
        transform.push({ translateY: slideAnim });
      }
    }

    if (scale) {
      transform.push({ scale: scaleAnim });
    }

    return {
      opacity: fadeAnim,
      transform,
    } as ViewStyle;
  }, [fadeAnim, slideAnim, scaleAnim, direction, slideDistance, scale]);

  return { fadeAnim, slideAnim, scaleAnim, animatedStyle, start };
}

interface PressAnimationOptions {
  scale?: number;
  tension?: number;
  friction?: number;
  duration?: number;
}

/**
 * 按压动画 Hook
 */
export function usePressAnimation(options: PressAnimationOptions = {}) {
  const {
    scale = 0.95,
    tension = 100,
    friction = 8,
  } = options;

  const scaleAnim = useRef(new Animated.Value(1)).current;
  
  const onPressIn = useCallback(() => {
    Animated.spring(scaleAnim, {
      toValue: scale,
      useNativeDriver: true,
      tension,
      friction,
    }).start();
  }, [scaleAnim, scale, tension, friction]);

  const onPressOut = useCallback(() => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension,
      friction,
    }).start();
  }, [scaleAnim, tension, friction]);

  const animatedStyle = useMemo(() => ({
    transform: [{ scale: scaleAnim }],
  }), [scaleAnim]);

  return { scaleAnim, onPressIn, onPressOut, animatedStyle };
}
