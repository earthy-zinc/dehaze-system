import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '../styles';

interface InfoRowProps {
  label: string;
  value: string;
  mono?: boolean;
}

/** 信息行 */
const InfoRow: React.FC<InfoRowProps> = ({ label, value, mono }) => (
  <View style={styles.infoRow}>
    <Text style={styles.infoLabel}>{label}</Text>
    <Text
      style={[styles.infoValue, mono && styles.infoValueMono]}
      numberOfLines={mono ? 2 : 1}
    >
      {value}
    </Text>
  </View>
);

export default InfoRow;