import React from 'react';
import {
  TouchableOpacity,
  Text,
  View,
  StyleSheet,
  ViewStyle,
  ActivityIndicator,
} from 'react-native';
// import LinearGradient from 'react-native-linear-gradient';

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
      return <ActivityIndicator color="#ffffff" size="small" />;
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
    borderRadius: 12,
    overflow: 'hidden',
  },
  gradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
    paddingVertical: 16,
    backgroundColor: '#3b82f6',
  },
  primary: {
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  primaryGradient: {
    paddingHorizontal: 40,
    paddingVertical: 16,
  },
  secondary: {
    backgroundColor: '#ffffff',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    paddingHorizontal: 40,
    paddingVertical: 16,
  },
  large: {
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 12,
  },
  largeGradient: {
    paddingHorizontal: 60,
    paddingVertical: 20,
  },
  disabled: {
    opacity: 0.5,
  },
  buttonText: {
    fontSize: 17,
    fontWeight: '600',
    textAlign: 'center',
  },
  primaryText: {
    color: '#ffffff',
  },
  secondaryText: {
    color: '#3b82f6',
  },
  largeText: {
    fontSize: 20,
    fontWeight: '700',
    color: '#ffffff',
  },
  iconContainer: {
    marginLeft: 8,
  },
});

export default Button;