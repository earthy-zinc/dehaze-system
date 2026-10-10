/**
 * 去雾模型选择器（内联于处理页，不再独立成页）
 *
 * 展示启用的 image_edit 模型，点击即切换当前模型。
 */
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import type { AiModelVO } from 'dehaze-sdk-js';

interface ModelPickerProps {
  models: AiModelVO[];
  loading: boolean;
  selectedModelId?: string;
  onSelect: (modelId: string) => void;
}

const ModelPicker: React.FC<ModelPickerProps> = ({
  models,
  loading,
  selectedModelId,
  onSelect,
}) => {
  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="small" color={theme.colors.primary} />
        <Text style={styles.hintText}>加载模型中...</Text>
      </View>
    );
  }

  if (models.length === 0) {
    return <Text style={styles.hintText}>暂无可用去雾模型</Text>;
  }

  return (
    <View style={styles.container}>
      {models.map(model => {
        const isSelected = model.modelId === selectedModelId;
        return (
          <TouchableOpacity
            key={model.modelId}
            style={[styles.modelCard, isSelected && styles.modelCardSelected]}
            onPress={() => onSelect(model.modelId)}
            activeOpacity={0.7}
          >
            <View style={styles.modelInfo}>
              <Text style={styles.modelName}>{model.displayName}</Text>
              <Text style={styles.modelId} numberOfLines={1}>
                {model.modelId}
              </Text>
            </View>
            {isSelected && (
              <Icon
                name="check-circle"
                size={20}
                color={theme.colors.primary}
              />
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: theme.spacing.sm,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
  },
  modelCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.background.primary,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  modelCardSelected: {
    borderColor: theme.colors.primary,
    backgroundColor: `${theme.colors.primary}10`,
  },
  modelInfo: {
    flex: 1,
    gap: 2,
  },
  modelName: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  modelId: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  hintText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginBottom: theme.spacing.sm,
  },
});

export default ModelPicker;
