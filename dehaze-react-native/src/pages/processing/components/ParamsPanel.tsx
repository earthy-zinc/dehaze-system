/**
 * 参数调节面板
 *
 * 提供：
 * - 预设方案选择（推荐/风景/夜景）
 * - 通用参数滑块（去雾强度/饱和度/对比度/锐化）
 * - 重置为默认值
 */
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import type { CommonAlgorithmParams } from '@/types/processing';
import {
  PARAM_SCHEMAS,
  PARAM_PRESETS,
  DEFAULT_PARAMS,
} from '../services/processingApi';

// RN 自带 Slider 在 0.71+ 已移除，使用社区包；若未安装则用简化版按钮调节
// 这里采用纯 TouchableOpacity + 数值显示的简化实现，避免引入额外依赖
interface ParamsPanelProps {
  params: CommonAlgorithmParams;
  onChange: (params: CommonAlgorithmParams) => void;
  disabled?: boolean;
}

const ParamsPanel: React.FC<ParamsPanelProps> = ({ params, onChange, disabled = false }) => {
  const handlePreset = (presetParams: CommonAlgorithmParams) => {
    if (disabled) return;
    onChange({ ...presetParams });
  };

  const handleReset = () => {
    if (disabled) return;
    onChange({ ...DEFAULT_PARAMS });
  };

  const handleStepChange = (key: keyof CommonAlgorithmParams, delta: number) => {
    if (disabled) return;
    const schema = PARAM_SCHEMAS.find(s => s.key === key);
    if (!schema) return;
    const current = params[key] ?? schema.default;
    const next = Math.max(
      schema.min ?? 0,
      Math.min(schema.max ?? 100, current + delta * (schema.step ?? 1)),
    );
    onChange({ ...params, [key]: next });
  };

  const handleSliderChange = (key: keyof CommonAlgorithmParams, value: number) => {
    if (disabled) return;
    onChange({ ...params, [key]: value });
  };

  return (
    <View style={styles.container}>
      {/* 预设方案 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>预设方案</Text>
        <View style={styles.presetRow}>
          {PARAM_PRESETS.map(preset => {
            const isActive = PARAM_SCHEMAS.every(
              s => params[s.key] === preset.params[s.key],
            );
            return (
              <TouchableOpacity
                key={preset.key}
                style={[
                  styles.presetChip,
                  isActive && styles.presetChipActive,
                  disabled && styles.disabled,
                ]}
                onPress={() => handlePreset(preset.params)}
                disabled={disabled}
              >
                <Text
                  style={[
                    styles.presetText,
                    isActive && styles.presetTextActive,
                  ]}
                >
                  {preset.name}
                </Text>
              </TouchableOpacity>
            );
          })}
          <TouchableOpacity
            style={[styles.resetButton, disabled && styles.disabled]}
            onPress={handleReset}
            disabled={disabled}
          >
            <Icon name="refresh" size={12} color={theme.colors.text.secondary} />
            <Text style={styles.resetText}>重置</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* 参数滑块 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>参数调节</Text>
        {PARAM_SCHEMAS.map(schema => {
          const value = params[schema.key] ?? schema.default;
          const fillPercent = ((value - (schema.min ?? 0)) / ((schema.max ?? 100) - (schema.min ?? 0))) * 100;
          return (
            <View key={schema.key} style={styles.paramItem}>
              <View style={styles.paramHeader}>
                <View style={styles.paramLabelRow}>
                  <Text style={styles.paramLabel}>{schema.label}</Text>
                  {schema.description && (
                    <Text style={styles.paramDesc} numberOfLines={1}>
                      {schema.description}
                    </Text>
                  )}
                </View>
                <Text style={styles.paramValue}>{value}</Text>
              </View>

              {/* 自定义滑块（进度条 + 步进按钮） */}
              <View style={styles.sliderRow}>
                <TouchableOpacity
                  onPress={() => handleStepChange(schema.key, -1)}
                  disabled={disabled}
                  style={[styles.stepButton, disabled && styles.disabled]}
                >
                  <Text style={styles.stepButtonText}>-</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.sliderTrack, disabled && styles.disabled]}
                  onPress={e => {
                    if (disabled) return;
                    // 简化：根据触摸位置计算值
                    const layout = e.nativeEvent.locationX;
                    const trackWidth = 200; // 近似值
                    const percent = Math.max(0, Math.min(1, layout / trackWidth));
                    const next = Math.round(
                      (schema.min ?? 0) + percent * ((schema.max ?? 100) - (schema.min ?? 0)),
                    );
                    handleSliderChange(schema.key, next);
                  }}
                  disabled={disabled}
                >
                  <View
                    style={[
                      styles.sliderFill,
                      { width: `${fillPercent}%` },
                    ]}
                  />
                  <View
                    style={[
                      styles.sliderThumb,
                      { left: `${fillPercent}%` },
                    ]}
                  />
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={() => handleStepChange(schema.key, 1)}
                  disabled={disabled}
                  style={[styles.stepButton, disabled && styles.disabled]}
                >
                  <Text style={styles.stepButtonText}>+</Text>
                </TouchableOpacity>
              </View>

              <View style={styles.paramRangeRow}>
                <Text style={styles.paramRange}>{schema.min}</Text>
                <Text style={styles.paramRange}>{schema.max}</Text>
              </View>
            </View>
          );
        })}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
  },
  section: {
    marginBottom: theme.spacing.md,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  presetRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  presetChip: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  presetChipActive: {
    backgroundColor: `${theme.colors.primary}15`,
    borderColor: theme.colors.primary,
  },
  presetText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  presetTextActive: {
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: theme.spacing.xs,
    borderRadius: theme.layout.borderRadius.full,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  resetText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  paramItem: {
    marginBottom: theme.spacing.md,
  },
  paramHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.xs,
  },
  paramLabelRow: {
    flex: 1,
    gap: 2,
  },
  paramLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  paramDesc: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  paramValue: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
    minWidth: 40,
    textAlign: 'right',
  },
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
  paramRangeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  paramRange: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  disabled: {
    opacity: 0.5,
  },
});

export default ParamsPanel;
