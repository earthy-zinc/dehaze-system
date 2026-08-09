import React, { useRef, useEffect } from 'react';
import { View, Text, StyleSheet, Animated, Easing } from 'react-native';
import Card from '@/components/Card';
import Icon from '@/components/Icon';
import { colors } from '@/theme/colors';

interface SpecCardProps {
  icon: string;
  title: string;
  description: string;
  compact?: boolean;
}

const SpecCard: React.FC<SpecCardProps> = ({
  icon,
  title,
  description,
  compact = false,
}) => {
  const scaleAnim = useRef(new Animated.Value(0.9)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 600,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();
  }, [scaleAnim, fadeAnim]);

  return (
    <Animated.View style={{
      opacity: fadeAnim,
      transform: [{ scale: scaleAnim }],
    }}>
      <Card 
        padding={compact ? 20 : 32} 
        margin={0} 
        borderRadius={20}
        style={styles.card}
      >
        <View style={[
          styles.iconContainer,
          compact && styles.iconContainerCompact,
        ]}>
          <Icon
            name={icon}
            size={compact ? 24 : 32}
            color={colors.text.inverse}
            backgroundColor={colors.primary}
            borderRadius={compact ? 28 : 36}
          />
        </View>
        <Text style={[
          styles.title,
          compact && styles.titleCompact,
        ]}>
          {title}
        </Text>
        <Text style={[
          styles.description,
          compact && styles.descriptionCompact,
        ]}>
          {description}
        </Text>
      </Card>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  card: {
    borderWidth: 2,
    borderColor: colors.border.light,
  },
  iconContainer: {
    alignSelf: 'center',
    marginBottom: 24,
  },
  iconContainerCompact: {
    marginBottom: 16,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.secondary,
    marginBottom: 12,
    textAlign: 'center',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  titleCompact: {
    fontSize: 12,
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    color: colors.text.tertiary,
    lineHeight: 21,
    textAlign: 'center',
  },
  descriptionCompact: {
    fontSize: 12,
    lineHeight: 18,
  },
});

export default SpecCard;