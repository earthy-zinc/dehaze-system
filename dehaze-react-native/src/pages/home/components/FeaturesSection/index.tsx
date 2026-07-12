import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import WorkflowStep from './WorkflowStep';
import ToolCard from './ToolCard';
import Icon from '@/components/Icon';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';

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
  onTaskCenterPress: () => void;
}

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
  onTaskCenterPress,
}) => {
  const { width, isMobile, isTablet, spacing, containerPadding, fontScale } = useResponsive();

  // ... workflowSteps and tools definitions ...
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
    {
      icon: 'task',
      title: '任务中心',
      description: '查看和管理数据集导出、下载等异步任务',
      onPress: onTaskCenterPress,
    },
  ];

  // 计算工作流步骤卡片宽度
  const stepCardWidth = isMobile 
    ? width - containerPadding * 2 - 40 
    : Math.min(280, (width - containerPadding * 2 - spacing * 4) / 3);

  // 计算工具卡片宽度（响应式网格）
  const toolColumns = isMobile ? 1 : isTablet ? 2 : 3;
  const toolCardWidth = (width - containerPadding * 2 - spacing * (toolColumns - 1)) / toolColumns;

  // 响应式字体大小
  const titleFontSize = isMobile ? theme.typography.sizes.h3 : theme.typography.sizes.h1 * fontScale;
  const subtitleFontSize = isMobile ? theme.typography.sizes.body : theme.typography.sizes.h6 * fontScale;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={[styles.section, { paddingHorizontal: containerPadding }]}>
        <Text style={[styles.sectionTitle, { fontSize: titleFontSize }]}>
          强大的功能生态
        </Text>
        <Text style={[styles.sectionSubtitle, { fontSize: subtitleFontSize }]}>
          从输入到输出，每一步都精心设计
        </Text>
      </View>

      {/* Workflow Steps */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={[
          styles.workflowContainer,
          { paddingHorizontal: containerPadding },
        ]}
      >
        {workflowSteps.map((step, index) => (
          <View key={`workflow-${index}`} style={styles.stepWrapper}>
            <WorkflowStep {...step} width={stepCardWidth} />
            {index < workflowSteps.length - 1 && (
              <View style={[
                styles.arrowContainer,
                isMobile && styles.arrowContainerMobile,
              ]}>
                <Icon 
                  name={isMobile ? 'arrow-down' : 'arrow-right'} 
                  size={isMobile ? 20 : 24} 
                  color={theme.colors.border.light} 
                />
              </View>
            )}
          </View>
        ))}
      </ScrollView>

      {/* Tools Grid */}
      <View style={[
        styles.toolsSection,
        { paddingHorizontal: containerPadding },
      ]}>
        <View style={[
          styles.toolsGrid,
          { gap: spacing },
        ]}>
          {tools.map((tool, index) => (
            <View 
              key={`tool-${index}`} 
              style={[
                styles.toolWrapper,
                { width: toolCardWidth - (isMobile ? 0 : spacing / toolColumns) },
              ]}
            >
              <ToolCard {...tool} />
            </View>
          ))}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    paddingVertical: theme.spacing.huge,
  },
  section: {
    marginBottom: theme.spacing.xxxl,
  },
  sectionTitle: {
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
    letterSpacing: theme.typography.letterSpacing.normal,
    textAlign: 'center',
  },
  sectionSubtitle: {
    color: theme.colors.text.secondary,
    lineHeight: 28.8,
    textAlign: 'center',
  },
  workflowContainer: {
    gap: 20,
    alignItems: 'center',
    paddingBottom: theme.spacing.xxxl,
  },
  stepWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  arrowContainer: {
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: theme.spacing.md,
  },
  arrowContainerMobile: {
    marginHorizontal: theme.spacing.sm,
  },
  toolsSection: {
    paddingTop: 20,
  },
  toolsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  toolWrapper: {
    marginBottom: theme.spacing.lg,
  },
});

export default FeaturesSection;