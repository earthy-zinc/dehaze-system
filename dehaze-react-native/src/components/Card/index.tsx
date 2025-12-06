import React from 'react';
import {
  StyleSheet,
  ViewStyle,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { usePressAnimation } from '@/hooks/useAnimation';
import { theme } from '@/theme';

interface CardProps {
  children: React.ReactNode;
  style?: ViewStyle;
  onPress?: () => void;
  elevation?: boolean;
  padding?: number;
  margin?: number;
  borderRadius?: number;
}

const Card: React.FC<CardProps> = ({
  children,
  style,
  onPress,
  elevation = true,
  padding = theme.spacing.xl,
  margin = 0,
  borderRadius = theme.layout.borderRadius.xl,
}) => {
  const { animatedStyle, onPressIn, onPressOut } = usePressAnimation({ scale: 0.98 });

  const cardStyle = [
    styles.card,
    {
      padding,
      margin,
      borderRadius,
      ...(elevation ? theme.layout.shadows.md : {}),
    },
    style,
  ];

  const CardComponent = (
    <Animated.View
      style={[
        cardStyle,
        onPress && animatedStyle,
      ]}
    >
      {children}
    </Animated.View>
  );

  if (onPress) {
    return (
      <TouchableOpacity
        onPress={onPress}
        onPressIn={onPressIn}
        onPressOut={onPressOut}
        activeOpacity={1}
      >
        {CardComponent}
      </TouchableOpacity>
    );
  }

  return CardComponent;
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: theme.colors.background.primary,
    borderWidth: 2,
    borderColor: theme.colors.border.transparent,
  },
});

export default Card;