import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Card from '@/components/Card';
import Icon from '@/components/Icon';

interface ToolCardProps {
  icon: string;
  title: string;
  description: string;
  onPress: () => void;
}

const ToolCard: React.FC<ToolCardProps> = ({
  icon,
  title,
  description,
  onPress,
}) => {
  return (
    <Card onPress={onPress} padding={32} margin={0} borderRadius={20}>
      <View style={styles.iconWrapper}>
        <Icon
          name={icon}
          size={24}
          color="#3b82f6"
          backgroundColor="#eff6ff"
          borderRadius={14}
        />
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.description}>{description}</Text>
    </Card>
  );
};

const styles = StyleSheet.create({
  iconWrapper: {
    marginBottom: 20,
    alignSelf: 'flex-start',
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 12,
  },
  description: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 22.4,
  },
});

export default ToolCard;