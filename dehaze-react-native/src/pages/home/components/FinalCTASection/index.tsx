import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Button from '@/components/Button';
import Icon from '@/components/Icon';

interface FinalCTASectionProps {
  onStartPress: () => void;
}

const { width } = Dimensions.get('window');

const FinalCTASection: React.FC<FinalCTASectionProps> = ({ onStartPress }) => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.ctaTitle}>
          准备好体验专业级图像去雾了吗？
        </Text>
        <Text style={styles.ctaSubtitle}>
          立即开始，让您的图像重获清晰
        </Text>

        <Button
          title="开始使用"
          onPress={onStartPress}
          variant="large"
          icon={<Icon name="arrow-right" size={18} color="#ffffff" />}
          style={styles.ctaButton}
        />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
    paddingVertical: 100,
    minHeight: Dimensions.get('window').height * 0.5,
  },
  content: {
    alignItems: 'center',
    width: '100%',
    maxWidth: 600,
  },
  ctaTitle: {
    fontSize: 36,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 16,
    letterSpacing: -0.5,
    textAlign: 'center',
    lineHeight: 43.2,
  },
  ctaSubtitle: {
    fontSize: 18,
    color: '#6b7280',
    marginBottom: 40,
    textAlign: 'center',
    lineHeight: 28.8,
  },
  ctaButton: {
    width: width > 400 ? 240 : width - 80,
  },
});

export default FinalCTASection;