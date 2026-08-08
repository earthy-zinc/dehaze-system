/**
 * 帮助中心 (L2)
 *
 * FAQ 折叠列表（静态）
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

interface FAQ {
  q: string;
  a: string;
}

const FAQ_LIST: FAQ[] = [
  {
    q: '如何使用去雾功能？',
    a: '进入「去雾」Tab，上传图片后选择算法，调节参数即可开始处理。处理完成后可在效果对比中查看结果。',
  },
  {
    q: '支持哪些图片格式？',
    a: '目前支持 JPG、PNG、BMP、WebP 等常见格式，单张图片大小建议不超过 20MB。',
  },
  {
    q: '如何查看处理历史？',
    a: '进入「我的」→「个人数据」→「处理历史」，可查看所有历史任务，支持按状态筛选和下载结果。',
  },
  {
    q: '会员权益有哪些？',
    a: 'VIP 会员享有更高的月度处理额度、批量处理、高清导出、报告导出等权益。进入「我的」→「我的会员」查看详情。',
  },
  {
    q: '如何反馈问题？',
    a: '进入「我的」→「反馈评价」，点击右下角「+」按钮提交反馈，我们会在 3 个工作日内回复。',
  },
  {
    q: '处理失败怎么办？',
    a: '处理失败可能是图片格式不支持或算法异常导致，建议更换图片或切换算法重试。如问题持续，请提交反馈。',
  },
];

const PersonalHelpScreen: React.FC = () => {
  const navigation = useNavigation();
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const toggleExpand = useCallback((idx: number) => {
    setExpandedIdx(prev => (prev === idx ? null : idx));
  }, []);

  return (
    <View style={styles.container}>
      <AppHeader title="帮助中心" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView contentContainerStyle={styles.content}>
      <Text style={styles.header}>常见问题</Text>

      {FAQ_LIST.map((faq, idx) => {
        const isExpanded = expandedIdx === idx;
        return (
          <TouchableOpacity
            key={idx}
            style={[styles.faqCard, isExpanded && styles.faqCardExpanded]}
            onPress={() => toggleExpand(idx)}
            activeOpacity={0.7}
          >
            <View style={styles.faqHeader}>
              <Text style={styles.faqQuestion}>{faq.q}</Text>
              <Ionicons
                name={isExpanded ? 'chevron-up' : 'chevron-down'}
                size={18}
                color={theme.colors.text.tertiary}
              />
            </View>
            {isExpanded && (
              <Text style={styles.faqAnswer}>{faq.a}</Text>
            )}
          </TouchableOpacity>
        );
      })}

      <View style={styles.contactSection}>
        <Text style={styles.contactTitle}>联系我们</Text>
        <Text style={styles.contactText}>
          如有其他问题，请通过「我的」→「反馈评价」提交反馈，或发送邮件至 support@dehaze.example.com
        </Text>
      </View>
    </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  header: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
  },
  faqCard: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  faqCardExpanded: {
    borderLeftWidth: 3,
    borderLeftColor: theme.colors.primary,
  },
  faqHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  faqQuestion: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginRight: theme.spacing.sm,
  },
  faqAnswer: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 20,
    marginTop: theme.spacing.sm,
    paddingTop: theme.spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: theme.colors.border.light,
  },
  contactSection: {
    marginTop: theme.spacing.xl,
    padding: theme.spacing.md,
  },
  contactTitle: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  contactText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 20,
  },
});

export default PersonalHelpScreen;
