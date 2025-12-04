import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface BadgeProps {
  text: string;
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'info' | 'foggy' | 'clear' | 'annotated';
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
        return { backgroundColor: '#3b82f6', color: '#ffffff' };
      case 'secondary':
        return { backgroundColor: '#f3f4f6', color: '#6b7280' };
      case 'success':
        return { backgroundColor: '#10b981', color: '#ffffff' };
      case 'warning':
        return { backgroundColor: '#f59e0b', color: '#ffffff' };
      case 'info':
        return { backgroundColor: '#06b6d4', color: '#ffffff' };
      case 'foggy':
        return { backgroundColor: '#6b7280', color: '#ffffff' };
      case 'clear':
        return { backgroundColor: '#3b82f6', color: '#ffffff' };
      case 'annotated':
        return { backgroundColor: '#10b981', color: '#ffffff' };
      default:
        return { backgroundColor: '#3b82f6', color: '#ffffff' };
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