/**
 * 通用滑块控制组件
 *
 * 提供步进按钮（- / +）、点击轨道、拖拽手势三种交互方式。
 * 拖拽通过 PanResponder 实现：触摸时立即更新值，移动时持续更新。
 */
import React, { useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  PanResponder,
  LayoutChangeEvent,
} from 'react-native';
import { theme } from '@/theme';

interface SliderControlProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

const SliderControl: React.FC<SliderControlProps> = ({
  value,
  min,
  max,
  step,
  onChange,
  disabled = false,
}) => {
  const [trackWidth, setTrackWidth] = useState(0);

  // 使用 ref 保存最新值，避免 PanResponder 闭包捕获陈旧值
  const trackWidthRef = useRef(0);
  trackWidthRef.current = trackWidth;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const disabledRef = useRef(disabled);
  disabledRef.current = disabled;
  const valueRef = useRef(value);
  valueRef.current = value;
  const rangeRef = useRef({ min, max, step });
  rangeRef.current = { min, max, step };

  const handleLayout = (e: LayoutChangeEvent) => {
    setTrackWidth(e.nativeEvent.layout.width);
  };

  /** 根据触摸 X 坐标计算新值（按 step 取整，clamp 到 [min, max]） */
  const calcValue = (touchX: number) => {
    const { min: lo, max: hi, step: st } = rangeRef.current;
    const width = trackWidthRef.current;
    if (width === 0 || hi <= lo) return valueRef.current;
    const percent = Math.max(0, Math.min(1, touchX / width));
    const raw = lo + percent * (hi - lo);
    const stepped = Math.round((raw - lo) / st) * st + lo;
    return Math.max(lo, Math.min(hi, stepped));
  };

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => !disabledRef.current,
      onMoveShouldSetPanResponder: () => !disabledRef.current,
      onPanResponderGrant: (evt) => {
        const next = calcValue(evt.nativeEvent.locationX);
        if (next !== valueRef.current) onChangeRef.current(next);
      },
      onPanResponderMove: (evt) => {
        const next = calcValue(evt.nativeEvent.locationX);
        if (next !== valueRef.current) onChangeRef.current(next);
      },
    }),
  ).current;

  const handleStep = (delta: number) => {
    if (disabled) return;
    const next = Math.max(min, Math.min(max, value + delta * step));
    onChange(next);
  };

  const fillPercent = max > min ? ((value - min) / (max - min)) * 100 : 0;

  return (
    <View style={styles.sliderRow}>
      <TouchableOpacity
        onPress={() => handleStep(-1)}
        disabled={disabled}
        style={[styles.stepButton, disabled && styles.disabled]}
      >
        <Text style={styles.stepButtonText}>-</Text>
      </TouchableOpacity>

      <View
        style={[styles.sliderTrack, disabled && styles.disabled]}
        onLayout={handleLayout}
        {...panResponder.panHandlers}
      >
        <View style={[styles.sliderFill, { width: `${fillPercent}%` }]} />
        <View style={[styles.sliderThumb, { left: `${fillPercent}%` }]} />
      </View>

      <TouchableOpacity
        onPress={() => handleStep(1)}
        disabled={disabled}
        style={[styles.stepButton, disabled && styles.disabled]}
      >
        <Text style={styles.stepButtonText}>+</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  stepButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepButtonText: {
    fontSize: 18,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.secondary,
  },
  sliderTrack: {
    flex: 1,
    height: 32,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: theme.layout.borderRadius.full,
    position: 'relative',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  sliderFill: {
    position: 'absolute',
    height: '100%',
    backgroundColor: `${theme.colors.primary}40`,
    borderRadius: theme.layout.borderRadius.full,
  },
  sliderThumb: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: theme.colors.primary,
    marginLeft: -10,
    top: '50%',
    marginTop: -10,
  },
  disabled: {
    opacity: 0.5,
  },
});

export default SliderControl;
