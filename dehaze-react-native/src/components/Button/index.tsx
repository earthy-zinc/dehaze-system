import React from 'react';
import {
  TouchableOpacity,
  Text,
  View,
  StyleSheet,
  ViewStyle,
  ActivityIndicator,
} from 'react-native';
import { theme } from '@/theme';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'large';
  loading?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
  style?: ViewStyle;
}

const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  loading = false,
  disabled = false,
  icon = null,
  style,
}) => {
  const renderButtonContent = () => {
    if (loading) {
      return <ActivityIndicator color={theme.colors.text.inverse} size="small" />;
    }

    return (
      <>
        <Text style={[styles.buttonText, styles[`${variant}Text`]]}>
          {title}
        </Text>
        {icon && <View style={styles.iconContainer}>{icon}</View>}
      </>
    );
  };

  if (variant === 'primary' || variant === 'large') {
    return (
      <TouchableOpacity
        style={[styles.button, styles[variant], style, disabled && styles.disabled]}
        onPress={onPress}
        disabled={disabled || loading}
        activeOpacity={0.8}
      >
        <View style={[styles.gradient, styles[`${variant}Gradient`]]}>
          {renderButtonContent()}
        </View>
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity
      style={[styles.button, styles[variant], style, disabled && styles.disabled]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.8}
    >
      {renderButtonContent()}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: theme.layout.borderRadius.md,
    overflow: 'hidden',
  },
  gradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: theme.spacing.xxl,
    paddingVertical: theme.spacing.md,
    backgroundColor: theme.colors.primary,
  },
  primary: {
    ...theme.layout.shadows.md,
    shadowColor: theme.colors.primary,
  },
  primaryGradient: {
    paddingHorizontal: theme.spacing.xxl,
    paddingVertical: theme.spacing.md,
  },
  secondary: {
    backgroundColor: theme.colors.background.primary,
    borderWidth: 2,
    borderColor: theme.colors.border.light,
    paddingHorizontal: theme.spacing.xxl,
    paddingVertical: theme.spacing.md,
  },
  large: {
    ...theme.layout.shadows.lg,
    shadowColor: theme.colors.primary,
  },
  largeGradient: {
    paddingHorizontal: theme.spacing.xxxl,
    paddingVertical: 20,
  },
  disabled: {
    opacity: 0.5,
  },
  buttonText: {
    fontSize: 17,
    fontWeight: theme.typography.weights.semibold,
    textAlign: 'center',
  },
  primaryText: {
    color: theme.colors.text.inverse,
  },
  secondaryText: {
    color: theme.colors.primary,
  },
  largeText: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.inverse,
  },
  iconContainer: {
    marginLeft: theme.spacing.sm,
  },
});

export default Button;