import React from 'react';
import { View, Text } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import type { IoniconName } from '@/components/Icon';
import { theme } from '@/theme';
import { styles } from '../styles';

interface SectionTitleProps {
  icon: string;
  title: string;
}

/** 章节标题 */
const SectionTitle: React.FC<SectionTitleProps> = ({ icon, title }) => (
  <View style={styles.sectionTitleRow}>
    <View style={styles.sectionTitleIcon}>
      <Ionicons name={icon as IoniconName} size={16} color={theme.colors.primary} />
    </View>
    <Text style={styles.sectionTitleText}>{title}</Text>
  </View>
);

export default SectionTitle;