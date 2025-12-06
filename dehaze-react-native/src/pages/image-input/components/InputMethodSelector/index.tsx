/**
 * 输入方式选择器组件
 */

import React, { useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
} from 'react-native';
import Icon from '@/components/Icon';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';
import { InputMethod, InputMethodConfig } from '../../types/imageInput';

// 输入方式配置
const INPUT_METHODS: InputMethodConfig[] = [
  {
    key: 'upload',
    icon: 'cloud-upload',
    title: '上传图片',
    subtitle: '从相册选择',
  },
  {
    key: 'camera',
    icon: 'camera',
    title: '拍照',
    subtitle: '实时拍摄',
  },
  {
    key: 'sample',
    icon: 'images',
    title: '样例图片',
    subtitle: '快速体验',
  },
  {
    key: 'history',
    icon: 'clock',
    title: '历史记录',
    subtitle: '最近处理',
  },
];

interface InputMethodSelectorProps {
  currentMethod: InputMethod;
  onMethodChange: (method: InputMethod) => void;
}

interface MethodButtonProps {
  config: InputMethodConfig;
  isActive: boolean;
  onPress: () => void;
}

const MethodButton: React.FC<MethodButtonProps> = ({ config, isActive, onPress }) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.95,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  return (
    <TouchableOpacity
      onPress={onPress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
      style={styles.buttonWrapper}
    >
      <Animated.View
        style={[
          styles.methodButton,
          isActive && styles.methodButtonActive,
          { transform: [{ scale: scaleAnim }] },
        ]}
      >
        <Icon
          name={config.icon}
          size={28}
          color={isActive ? theme.colors.primary : theme.colors.text.secondary}
        />
        <Text
          style={[
            styles.methodTitle,
            isActive && styles.methodTitleActive,
          ]}
        >
          {config.title}
        </Text>
        <Text
          style={[
            styles.methodSubtitle,
            isActive && styles.methodSubtitleActive,
          ]}
        >
          {config.subtitle}
        </Text>
      </Animated.View>
    </TouchableOpacity>
  );
};

const InputMethodSelector: React.FC<InputMethodSelectorProps> = ({
  currentMethod,
  onMethodChange,
}) => {
  const { isMobile } = useResponsive();

  return (
    <View style={styles.container}>
      <View style={[styles.grid, !isMobile && styles.gridDesktop]}>
        {INPUT_METHODS.map(config => (
          <MethodButton
            key={config.key}
            config={config}
            isActive={currentMethod === config.key}
            onPress={() => onMethodChange(config.key)}
          />
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: theme.spacing.lg,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -6,
  },
  gridDesktop: {
    flexWrap: 'nowrap',
  },
  buttonWrapper: {
    width: '50%',
    padding: 6,
  },
  methodButton: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: theme.colors.border.light,
    ...theme.layout.shadows.sm,
  },
  methodButtonActive: {
    borderColor: theme.colors.primary,
    backgroundColor: `${theme.colors.primary}10`,
  },
  methodTitle: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginTop: theme.spacing.sm,
  },
  methodTitleActive: {
    color: theme.colors.primary,
  },
  methodSubtitle: {
    fontSize: theme.typography.sizes.caption,
    color: theme.colors.text.tertiary,
    marginTop: 2,
  },
  methodSubtitleActive: {
    color: theme.colors.primary,
    opacity: 0.8,
  },
});

export default InputMethodSelector;
