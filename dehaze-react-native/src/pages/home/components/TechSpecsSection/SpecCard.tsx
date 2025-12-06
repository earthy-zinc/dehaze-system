import React, { useRef, useEffect } from 'react';
import { View, Text, StyleSheet, Animated, Easing } from 'react-native';
import Card from '@/components/Card';
import Icon from '@/components/Icon';

interface SpecCardProps {
  icon: string;
  title: string;
  value: string;
  description: string;
  compact?: boolean;
}

const SpecCard: React.FC<SpecCardProps> = ({
  icon,
  title,
  value,
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
            color="#ffffff"
            backgroundColor="#3b82f6"
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
          styles.value,
          compact && styles.valueCompact,
        ]}>
          {value}
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
    borderColor: '#e5e7eb',
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
    color: '#6b7280',
    marginBottom: 12,
    textAlign: 'center',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  titleCompact: {
    fontSize: 12,
    marginBottom: 8,
  },
  value: {
    fontSize: 48,
    fontWeight: '700',
    color: '#3b82f6',
    marginBottom: 8,
    textAlign: 'center',
    lineHeight: 48,
  },
  valueCompact: {
    fontSize: 32,
    lineHeight: 36,
  },
  description: {
    fontSize: 14,
    color: '#9ca3af',
    lineHeight: 21,
    textAlign: 'center',
  },
  descriptionCompact: {
    fontSize: 12,
    lineHeight: 18,
  },
});

export default SpecCard;