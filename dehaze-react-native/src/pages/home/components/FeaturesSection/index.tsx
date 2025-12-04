import React from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions } from 'react-native';
import WorkflowStep from './WorkflowStep';
import ToolCard from './ToolCard';
import Icon from '@/components/Icon';

interface FeaturesSectionProps {
  onImageInputPress: () => void;
  onAlgorithmSelectPress: () => void;
  onProcessingPress: () => void;
  onSideBySidePress: () => void;
  onOverlayPress: () => void;
  onMagnifierPress: () => void;
  onFilterPress: () => void;
  onMetricsPress: () => void;
  onDatasetManagePress: () => void;
}

const { width } = Dimensions.get('window');

const FeaturesSection: React.FC<FeaturesSectionProps> = ({
  onImageInputPress,
  onAlgorithmSelectPress,
  onProcessingPress,
  onSideBySidePress,
  onOverlayPress,
  onMagnifierPress,
  onFilterPress,
  onMetricsPress,
  onDatasetManagePress,
}) => {
  const workflowSteps = [
    {
      number: '01',
      icon: 'image',
      title: '图像输入',
      description: '支持上传、拍照、样例图片\n多种输入方式随心选择',
      onPress: onImageInputPress,
    },
    {
      number: '02',
      icon: 'brain',
      title: '智能算法',
      description: '多种去雾算法可选\nAI智能推荐最优方案',
      onPress: onAlgorithmSelectPress,
    },
    {
      number: '03',
      icon: 'magic',
      title: '一键处理',
      description: '毫秒级处理速度\n实时预览处理效果',
      onPress: onProcessingPress,
    },
  ];

  const tools = [
    {
      icon: 'columns',
      title: '并排对比',
      description: '多图并排展示，支持2-4张图片同屏对比',
      onPress: onSideBySidePress,
    },
    {
      icon: 'layer-group',
      title: '重叠对比',
      description: '拖动分割线实时对比，支持横向和纵向模式',
      onPress: onOverlayPress,
    },
    {
      icon: 'search-plus',
      title: '放大镜',
      description: '局部细节放大查看，精确对比图像质量',
      onPress: onMagnifierPress,
    },
    {
      icon: 'sliders-h',
      title: '滤镜调节',
      description: '实时调节亮度、对比度、饱和度等参数',
      onPress: onFilterPress,
    },
    {
      icon: 'chart-line',
      title: '指标评估',
      description: 'SSIM、PSNR等专业指标定量分析',
      onPress: onMetricsPress,
    },
    {
      icon: 'database',
      title: '数据集管理',
      description: '浏览和管理多个专业去雾数据集',
      onPress: onDatasetManagePress,
    },
  ];

  return (
    <View style={styles.container}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>强大的功能生态</Text>
        <Text style={styles.sectionSubtitle}>从输入到输出，每一步都精心设计</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.workflowContainer}
        >
          {workflowSteps.map((step, index) => (
            <View key={`workflow-${index}`} style={styles.stepWrapper}>
              <WorkflowStep {...step} />
              {index < workflowSteps.length - 1 && (
                <View style={styles.arrowContainer}>
                  <Icon name="arrow-right" size={24} color="#d1d5db" />
                </View>
              )}
            </View>
          ))}
        </ScrollView>
      </View>

      <View style={styles.toolsSection}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.toolsContainer}
        >
          {tools.map((tool, index) => (
            <View key={`tool-${index}`} style={styles.toolWrapper}>
              <ToolCard {...tool} />
            </View>
          ))}
        </ScrollView>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  section: {
    paddingHorizontal: 20,
    paddingVertical: 80,
  },
  sectionTitle: {
    fontSize: 40,
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: 16,
    letterSpacing: -0.5,
    textAlign: 'center',
  },
  sectionSubtitle: {
    fontSize: 18,
    color: '#6b7280',
    lineHeight: 28.8,
    textAlign: 'center',
    marginBottom: 60,
  },
  workflowContainer: {
    paddingHorizontal: 20,
    gap: 20,
    alignItems: 'center',
  },
  stepWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 20,
  },
  arrowContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  toolsSection: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  toolsContainer: {
    gap: 24,
  },
  toolWrapper: {
    width: width - 40,
  },
});

export default FeaturesSection;