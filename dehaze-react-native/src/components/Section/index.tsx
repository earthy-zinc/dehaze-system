import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ViewStyle,
} from 'react-native';

interface SectionProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  style?: ViewStyle;
  contentStyle?: ViewStyle;
  centered?: boolean;
  padding?: number;
}

const Section: React.FC<SectionProps> = ({
  title,
  subtitle,
  children,
  style,
  contentStyle,
  centered = true,
  padding = 80,
}) => {

  return (
    <View style={[styles.container, { padding }, style]}>
      {(title || subtitle) && (
        <View style={[styles.header, centered && styles.centeredHeader]}>
          {title && <Text style={styles.title}>{title}</Text>}
          {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
        </View>
      )}
      <View style={[styles.content, contentStyle]}>{children}</View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    maxWidth: '100%',
  },
  header: {
    marginBottom: 60,
  },
  centeredHeader: {
    alignItems: 'center',
  },
  title: {
    fontSize: 40,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 16,
    letterSpacing: -0.5,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    color: '#6b7280',
    lineHeight: 28.8,
    textAlign: 'center',
  },
  content: {
    flex: 1,
  },
});

export default Section;