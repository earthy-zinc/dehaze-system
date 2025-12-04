import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Button from '@/components/Button';
import Icon from '@/components/Icon';
import Card from '@/components/Card';

interface AlgorithmSectionProps {
  onLearnMorePress: () => void;
}

const AlgorithmSection: React.FC<AlgorithmSectionProps> = ({
  onLearnMorePress,
}) => {
  const algorithmFeatures = [
    '智能推荐最适合的去雾算法',
    '实时对比不同算法的处理效果',
    '毫秒级处理速度，即时查看结果',
    '支持批量处理和参数自定义',
  ];

  const algorithmImageUrl = 'https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/f49e4b9e-6079-4a0b-8f91-bcab5deec2c7/image_1763727581_1_3.jpg';

  return (
    <View style={styles.container}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <View style={styles.algorithmContent}>
          <View style={styles.algorithmText}>
            <Text style={styles.sectionTitle}>多算法智能选择</Text>
            <Text style={styles.sectionSubtitle}>
              支持DCP、AOD-Net、DehazeNet等多种先进算法
            </Text>

            <View style={styles.featuresList}>
              {algorithmFeatures.map((feature, index) => (
                <View key={`feature-${index}`} style={styles.featureItem}>
                  <Icon
                    name="check-circle"
                    size={20}
                    color="#34d399"
                    style={styles.featureIcon}
                  />
                  <Text style={styles.featureText}>{feature}</Text>
                </View>
              ))}
            </View>

            <Button
              title="了解更多算法详情"
              onPress={onLearnMorePress}
              variant="secondary"
              icon={<Icon name="arrow-right" size={14} color="#3b82f6" />}
              style={styles.learnMoreButton}
            />
          </View>

          <View style={styles.algorithmVisual}>
            <Card padding={0} margin={0} borderRadius={20}>
              <ImageLoader
                source={{ uri: algorithmImageUrl }}
                style={styles.algorithmImage}
                containerStyle={styles.imageContainer}
              />
            </Card>
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    backgroundColor: '#3b82f6',
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingVertical: 80,
  },
  algorithmContent: {
    flexDirection: 'column',
    gap: 40,
  },
  algorithmText: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 40,
    fontWeight: '700',
    color: '#ffffff',
    marginBottom: 16,
    letterSpacing: -0.5,
  },
  sectionSubtitle: {
    fontSize: 18,
    color: 'rgba(255, 255, 255, 0.8)',
    lineHeight: 28.8,
    marginBottom: 32,
  },
  featuresList: {
    marginBottom: 32,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    marginBottom: 16,
  },
  featureIcon: {
    marginTop: 2,
  },
  featureText: {
    flex: 1,
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.95)',
    lineHeight: 25.6,
  },
  learnMoreButton: {
    alignSelf: 'flex-start',
  },
  algorithmVisual: {
    flex: 1,
  },
  imageContainer: {
    width: '100%',
    height: 240,
    borderRadius: 20,
    overflow: 'hidden',
  },
  algorithmImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
});

export default AlgorithmSection;