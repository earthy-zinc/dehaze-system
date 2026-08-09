import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '../styles';

interface MetricBarProps {
  label: string;
  value: number;
  max: number;
  color: string;
  suffix?: string;
  precision?: number;
}

/** 指标条 */
const MetricBar: React.FC<MetricBarProps> = ({
  label,
  value,
  max,
  color,
  suffix = '',
  precision = 0,
}) => {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <View style={styles.metricBarWrap}>
      <View style={styles.metricBarHeader}>
        <Text style={styles.metricLabel}>{label}</Text>
        <Text style={[styles.metricValue, { color }]}>
          {value.toFixed(precision)}
          <Text style={styles.metricSuffix}>{suffix}</Text>
        </Text>
      </View>
      <View style={styles.metricBarTrack}>
        <View
          style={[
            styles.metricBarFill,
            { width: `${pct}%`, backgroundColor: color },
          ]}
        />
      </View>
    </View>
  );
};

export default MetricBar;