import React from 'react';
import {
  StyleSheet,
  ViewStyle,
  TouchableOpacity,
  Animated,
} from 'react-native';

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
  padding = 32,
  margin = 0,
  borderRadius = 20,
}) => {
  const animatedValue = React.useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    if (onPress) {
      Animated.spring(animatedValue, {
        toValue: 0.98,
        useNativeDriver: true,
        tension: 100,
        friction: 8,
      }).start();
    }
  };

  const handlePressOut = () => {
    if (onPress) {
      Animated.spring(animatedValue, {
        toValue: 1,
        useNativeDriver: true,
        tension: 100,
        friction: 8,
      }).start();
    }
  };

  const cardStyle = [
    styles.card,
    {
      padding,
      margin,
      borderRadius,
      shadowColor: elevation ? '#000' : 'transparent',
      shadowOffset: elevation ? { width: 0, height: 4 } : { width: 0, height: 0 },
      shadowOpacity: elevation ? 0.06 : 0,
      shadowRadius: elevation ? 16 : 0,
      elevation: elevation ? 4 : 0,
    },
    style,
  ];

  const CardComponent = (
    <Animated.View
      style={[
        cardStyle,
        onPress && {
          transform: [{ scale: animatedValue }],
        },
      ]}
    >
      {children}
    </Animated.View>
  );

  if (onPress) {
    return (
      <TouchableOpacity
        onPress={onPress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
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
    backgroundColor: '#ffffff',
    borderWidth: 2,
    borderColor: 'transparent',
  },
});

export default Card;