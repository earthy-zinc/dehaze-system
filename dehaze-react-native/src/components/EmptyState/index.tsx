import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Icon from '../Icon';

interface EmptyStateProps {
  icon?: string;
  title: string;
  description?: string;
  iconSize?: number;
  iconColor?: string;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  icon = 'search-plus',
  title,
  description,
  iconSize = 60,
  iconColor = '#d1d5db',
}) => {
  return (
    <View style={styles.container}>
      <Icon
        name={icon}
        size={iconSize}
        color={iconColor}
        style={styles.icon}
      />
      <Text style={styles.title}>{title}</Text>
      {description && (
        <Text style={styles.description}>{description}</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
    paddingHorizontal: 40,
  },
  icon: {
    marginBottom: 16,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
    lineHeight: 20,
  },
});

export default EmptyState;