import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { theme } from '@/theme';

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
  padding = theme.spacing.huge,
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
    marginBottom: theme.spacing.xxxl,
  },
  centeredHeader: {
    alignItems: 'center',
  },
  title: {
    fontSize: theme.typography.sizes.h1,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
    letterSpacing: theme.typography.letterSpacing.normal,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: theme.typography.sizes.h6,
    color: theme.colors.text.secondary,
    lineHeight: theme.typography.sizes.h6 * theme.typography.lineHeights.body,
    textAlign: 'center',
  },
  content: {
    flex: 1,
  },
});

export default Section;