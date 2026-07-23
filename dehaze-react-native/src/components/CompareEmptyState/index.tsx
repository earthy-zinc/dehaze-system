/**
 * 对比页空状态组件
 *
 * 5 个对比页（SideBySide/Overlay/Magnifier/Filter/Metrics）在缺少必要参数时
 * 共用的空状态：图标 + 标题 + 描述 + "去选择图片"按钮。
 */
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { theme } from '@/theme';
import Icon from '@/components/Icon';

interface CompareEmptyStateProps {
  /** "去选择图片"按钮回调 */
  onPress: () => void;
}

const CompareEmptyState: React.FC<CompareEmptyStateProps> = ({ onPress }) => {
  return (
    <View style={styles.container}>
      <Icon name="image" size={48} color={theme.colors.text.tertiary} />
      <Text style={styles.title}>请先完成去雾处理</Text>
      <Text style={styles.desc}>对比功能需要先处理图片</Text>
      <TouchableOpacity style={styles.button} onPress={onPress}>
        <Text style={styles.buttonText}>去选择图片</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: theme.spacing.xl,
  },
  title: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginTop: theme.spacing.md,
    marginBottom: theme.spacing.xs,
  },
  desc: {
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.lg,
    textAlign: 'center',
  },
  button: {
    paddingHorizontal: theme.spacing.xl,
    paddingVertical: theme.spacing.md,
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.md,
  },
  buttonText: {
    color: '#fff',
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
  },
});

export default CompareEmptyState;
