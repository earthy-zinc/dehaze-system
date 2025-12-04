import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { ScrollView } from 'react-native-gesture-handler';
import Section from '@/components/Section';
import ImageLoader from '@/components/ImageLoader';
import Card from '@/components/Card';

interface ShowcaseSectionProps {
  onPress?: () => void;
}

const { width } = Dimensions.get('window');

const ShowcaseSection: React.FC<ShowcaseSectionProps> = ({ onPress }) => {
  const showcaseImageUrl = 'https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/20b8704f-d37e-45b9-a6c8-3c5d297e8a98/image_1763727568_3_3.jpg';

  return (
    <Section
      title="一键去雾，效果显著"
      subtitle="智能算法自动识别雾霾程度，精准还原图像细节"
      padding={80}
    >
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContainer}
      >
        <Card onPress={onPress} margin={0} padding={0} borderRadius={24}>
          <View style={styles.comparisonContainer}>
            <ImageLoader
              source={{ uri: showcaseImageUrl }}
              style={styles.showcaseImage}
              containerStyle={styles.imageContainer}
            />
            <View style={styles.comparisonLabel}>
              <View style={styles.labelItem}>
                <View style={[styles.labelDot, styles.beforeDot]} />
                <Text style={[styles.labelText, styles.beforeText]}>去雾前</Text>
              </View>
              <Text style={styles.divider}>→</Text>
              <View style={styles.labelItem}>
                <View style={[styles.labelDot, styles.afterDot]} />
                <Text style={[styles.labelText, styles.afterText]}>去雾后</Text>
              </View>
            </View>
          </View>
        </Card>
      </ScrollView>
    </Section>
  );
};

const styles = StyleSheet.create({
  scrollContainer: {
    paddingHorizontal: 20,
  },
  comparisonContainer: {
    position: 'relative',
    overflow: 'hidden',
  },
  imageContainer: {
    width: width - 40,
    height: 240,
    minHeight: 200,
  },
  showcaseImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  comparisonLabel: {
    position: 'absolute',
    bottom: 24,
    left: '50%',
    transform: [{ translateX: -0.5 * (width - 40) }],
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 100,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    width: width - 40,
    justifyContent: 'center',
  },
  labelItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  labelDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  beforeDot: {
    backgroundColor: '#fbbf24',
  },
  afterDot: {
    backgroundColor: '#34d399',
  },
  labelText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#ffffff',
  },
  beforeText: {
    color: '#fbbf24',
  },
  afterText: {
    color: '#34d399',
  },
  divider: {
    fontSize: 16,
    fontWeight: '600',
    color: '#9ca3af',
  },
});

export default ShowcaseSection;