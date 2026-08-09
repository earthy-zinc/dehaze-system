import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Card from '@/components/Card';
import Icon from '@/components/Icon';
import { colors } from '@/theme/colors';

interface WorkflowStepProps {
  number: string;
  icon: string;
  title: string;
  description: string;
  onPress: () => void;
  width?: number;
}

const WorkflowStep: React.FC<WorkflowStepProps> = ({
  number,
  icon,
  title,
  description,
  onPress,
  width = 280,
}) => {
  const cardStyle = width < 200 
    ? { ...styles.stepCard, ...styles.stepCardMinWidth, width } 
    : { ...styles.stepCard, width };

  return (
    <Card
      onPress={onPress}
      padding={32}
      margin={0}
      borderRadius={20}
      style={cardStyle}
    >
      <View style={styles.numberContainer}>
        <Text style={styles.number}>{number}</Text>
      </View>
      <View style={styles.iconContainer}>
        <Icon
          name={icon}
          size={28}
          color={colors.text.inverse}
          backgroundColor={colors.primary}
          borderRadius={16}
        />
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.description}>{description}</Text>
    </Card>
  );
};

const styles = StyleSheet.create({
  stepCard: {
    position: 'relative',
  },
  stepCardMinWidth: {
    minWidth: 200,
  },
  numberContainer: {
    position: 'absolute',
    top: 16,
    right: 16,
    width: 32,
    height: 32,
    backgroundColor: colors.primaryLight,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  number: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.primary,
  },
  iconContainer: {
    alignSelf: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 12,
    textAlign: 'center',
  },
  description: {
    fontSize: 14,
    color: colors.text.secondary,
    lineHeight: 22.4,
    textAlign: 'center',
  },
});

export default WorkflowStep;