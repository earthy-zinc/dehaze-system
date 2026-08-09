import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors } from '@/theme/colors';

interface BadgeProps {
  text: string;
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'info' | 'foggy';
  size?: 'small' | 'medium' | 'large';
  rounded?: boolean;
}

const Badge: React.FC<BadgeProps> = ({
  text,
  variant = 'primary',
  size = 'medium',
  rounded = true,
}) => {
  const getVariantStyle = () => {
    switch (variant) {
      case 'primary':
        return { backgroundColor: colors.primary, color: colors.text.inverse };
      case 'secondary':
        return { backgroundColor: colors.background.tertiary, color: colors.text.secondary };
      case 'success':
        return { backgroundColor: colors.badge.success.bg, color: colors.badge.success.text };
      case 'warning':
        return { backgroundColor: colors.badge.warning.bg, color: colors.badge.warning.text };
      case 'info':
        return { backgroundColor: colors.badge.info.bg, color: colors.badge.info.text };
      case 'foggy':
        return { backgroundColor: colors.text.secondary, color: colors.text.inverse };
      default:
        return { backgroundColor: colors.primary, color: colors.text.inverse };
    }
  };

  const getSizeStyle = () => {
    switch (size) {
      case 'small':
        return { paddingHorizontal: 6, paddingVertical: 2 };
      case 'medium':
        return { paddingHorizontal: 8, paddingVertical: 4 };
      case 'large':
        return { paddingHorizontal: 12, paddingVertical: 6 };
      default:
        return { paddingHorizontal: 8, paddingVertical: 4 };
    }
  };

  const getTextSizeStyle = () => {
    switch (size) {
      case 'small':
        return { fontSize: 10 };
      case 'medium':
        return { fontSize: 12 };
      case 'large':
        return { fontSize: 14 };
      default:
        return { fontSize: 12 };
    }
  };

  const variantStyle = getVariantStyle();
  const sizeStyle = getSizeStyle();
  const textStyle = getTextSizeStyle();

  return (
    <View
      style={[
        styles.badge,
        sizeStyle,
        { backgroundColor: variantStyle.backgroundColor },
        rounded && styles.rounded,
      ]}
    >
      <Text
        style={[
          styles.text,
          textStyle,
          { color: variantStyle.color },
        ]}
      >
        {text}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-start',
    borderWidth: 0,
  },
  rounded: {
    borderRadius: 12,
  },
  text: {
    fontWeight: '500',
    textAlign: 'center',
  },
});

export default Badge;