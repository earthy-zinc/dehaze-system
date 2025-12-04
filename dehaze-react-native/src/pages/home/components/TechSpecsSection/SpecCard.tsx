import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Card from '@/components/Card';
import Icon from '@/components/Icon';

interface SpecCardProps {
  icon: string;
  title: string;
  value: string;
  description: string;
}

const SpecCard: React.FC<SpecCardProps> = ({
  icon,
  title,
  value,
  description,
}) => {
  return (
    <Card padding={40} margin={0} borderRadius={20}>
      <View style={styles.iconContainer}>
        <Icon
          name={icon}
          size={32}
          color="#ffffff"
          backgroundColor="#3b82f6"
          borderRadius={36}
        />
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.value}>{value}</Text>
      <Text style={styles.description}>{description}</Text>
    </Card>
  );
};

const styles = StyleSheet.create({
  iconContainer: {
    alignSelf: 'center',
    marginBottom: 24,
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
  value: {
    fontSize: 48,
    fontWeight: '700',
    color: '#3b82f6',
    marginBottom: 8,
    textAlign: 'center',
    lineHeight: 48,
  },
  description: {
    fontSize: 14,
    color: '#9ca3af',
    lineHeight: 21,
    textAlign: 'center',
  },
});

export default SpecCard;