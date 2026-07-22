/**
 * 算法详情页面 (F-M03-004)
 *
 * 实现文档 `03-模块设计/核心模块/算法选择/需求规格.md` 中 F-M03-004 节定义的详情页结构：
 *  - 基本信息 / 算法描述 / 效果样例 / 参数配置 / 性能指标 / 用户评价 / 相关链接 / 版本历史
 *
 * 数据来源：
 *  - AlgorithmAPI.getAlgorithmInfoById(id) → 基本信息
 *  - AlgorithmAPI.getMonitorData(id)       → 性能指标
 *  - AlgorithmAPI.getVersions(id)          → 版本历史
 *
 * 移动端将文档描述的「左侧锚点 + 右侧内容」布局转换为：
 *  - 顶部横向滚动的章节锚点条
 *  - 单列滚动的章节内容
 *  - 底部固定的「立即使用 / 收藏 / 对比」操作栏
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  LayoutChangeEvent,
  Alert,
  Share,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import LinearGradient from 'react-native-linear-gradient';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import LoadingSpinner from '@/components/LoadingSpinner';
import EmptyState from '@/components/EmptyState';
import Badge from '@/components/Badge';

import { AlgorithmAPI } from 'dehaze-sdk-js';
import type {
  Algorithm,
  AlgorithmVersionVO,
  AlgorithmMonitorVO,
} from 'dehaze-sdk-js';
import AlgorithmSelectAPI from '@/api/algorithm-select';

type Props = NativeStackScreenProps<RootStackParamList, 'Algorithm'>;

/** 算法状态枚举映射 (来自后端 AlgorithmStatusEnum) */
const ALGORITHM_STATUS_MAP: Record<
  number,
  { label: string; color: string; bgColor: string }
> = {
  0: { label: '草稿', color: '#6B7280', bgColor: '#F3F4F6' },
  1: { label: '测试中', color: '#B45309', bgColor: '#FEF3C7' },
  2: { label: '待审核', color: '#1D4ED8', bgColor: '#DBEAFE' },
  3: { label: '已发布', color: '#047857', bgColor: '#D1FAE5' },
  4: { label: '已停用', color: '#B91C1C', bgColor: '#FEE2E2' },
  5: { label: '已归档', color: '#6B7280', bgColor: '#E5E7EB' },
};

/** 章节定义 */
interface SectionDef {
  key: string;
  label: string;
  icon: string;
}

const SECTIONS: SectionDef[] = [
  { key: 'basic', label: '基本信息', icon: 'information-circle' },
  { key: 'description', label: '算法描述', icon: 'document-text' },
  { key: 'samples', label: '效果样例', icon: 'images' },
  { key: 'params', label: '参数配置', icon: 'options' },
  { key: 'metrics', label: '性能指标', icon: 'analytics' },
  { key: 'reviews', label: '用户评价', icon: 'star' },
  { key: 'links', label: '相关链接', icon: 'link' },
  { key: 'versions', label: '版本历史', icon: 'git-branch' },
];

const AlgorithmScreen: React.FC<Props> = ({ route, navigation }) => {
  const algorithmId = route.params?.algorithmId;

  // 数据状态
  const [algorithm, setAlgorithm] = useState<Algorithm | null>(null);
  const [monitor, setMonitor] = useState<AlgorithmMonitorVO | null>(null);
  const [versions, setVersions] = useState<AlgorithmVersionVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // UI 状态
  const [activeSection, setActiveSection] = useState('basic');
  const [isFavorite, setIsFavorite] = useState(false);
  const [favoriteBusy, setFavoriteBusy] = useState(false);

  // 章节位置测量（用于点击锚点后滚动定位）
  const sectionLayouts = useRef<Record<string, number>>({});
  const scrollRef = useRef<ScrollView>(null);

  /** 加载算法详情 */
  const loadAlgorithm = useCallback(async () => {
    if (!algorithmId) {
      setError('缺少算法 ID');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const info = await AlgorithmAPI.getAlgorithmInfoById(algorithmId);
      setAlgorithm(info);
      // 并行加载监控数据与版本历史（失败不阻塞页面）
      Promise.allSettled([
        AlgorithmAPI.getMonitorData(algorithmId),
        AlgorithmAPI.getVersions(algorithmId),
      ]).then(([monRes, verRes]) => {
        if (monRes.status === 'fulfilled') setMonitor(monRes.value);
        if (verRes.status === 'fulfilled') setVersions(verRes.value || []);
      });
    } catch (e: any) {
      setError(e?.message || '加载算法详情失败');
    } finally {
      setLoading(false);
    }
  }, [algorithmId]);

  useEffect(() => {
    loadAlgorithm();
  }, [loadAlgorithm]);

  /** 同步收藏状态 */
  useEffect(() => {
    if (!algorithmId) return;
    AlgorithmSelectAPI.listFavorites()
      .then(list => {
        setIsFavorite(list.some(r => r.algorithmId === algorithmId));
      })
      .catch(() => {
        /* Python 后端未启动时静默忽略 */
      });
  }, [algorithmId]);

  /** 章节锚点点击 → 滚动到对应位置 */
  const handleSectionPress = useCallback((key: string) => {
    setActiveSection(key);
    const y = sectionLayouts.current[key];
    if (y !== undefined && scrollRef.current) {
      scrollRef.current.scrollTo({ y: y - 140, animated: true });
    }
  }, []);

  /** 测量章节位置 */
  const handleSectionLayout = useCallback(
    (key: string) => (e: LayoutChangeEvent) => {
      sectionLayouts.current[key] = e.nativeEvent.layout.y;
    },
    [],
  );

  /** 滚动时更新激活章节 */
  const handleScroll = useCallback(
    (event: any) => {
      const y = event.nativeEvent.contentOffset.y + 160;
      let current = activeSection;
      for (const section of SECTIONS) {
        const top = sectionLayouts.current[section.key];
        if (top !== undefined && top <= y) {
          current = section.key;
        }
      }
      if (current !== activeSection) {
        setActiveSection(current);
      }
    },
    [activeSection],
  );

  /** 立即使用 → 跳转处理页 */
  const handleUse = useCallback(() => {
    if (!algorithm) return;
    navigation.navigate('Processing', { algorithmId: algorithm.id });
  }, [algorithm, navigation]);

  /** 切换收藏 */
  const handleToggleFavorite = useCallback(() => {
    if (!algorithm) return;
    setFavoriteBusy(true);
    AlgorithmSelectAPI.toggleFavorite(algorithm.id)
      .then(res => {
        setIsFavorite(res.favorited);
      })
      .catch(err => {
        Alert.alert(
          '操作失败',
          err?.message || '收藏服务不可用，请稍后再试',
        );
      })
      .finally(() => setFavoriteBusy(false));
  }, [algorithm]);

  /** 分享算法 */
  const handleShare = useCallback(async () => {
    if (!algorithm) return;
    try {
      await Share.share({
        message: `推荐算法：${algorithm.name}\n类型：${algorithm.type}\n版本：v${algorithm.version || '1.0.0'}`,
      });
    } catch {
      /* 用户取消分享 */
    }
  }, [algorithm]);

  if (loading) {
    return (
      <MainLayout title="算法详情" showBack>
        <View style={styles.stateContainer}>
          <LoadingSpinner size="large" color={theme.colors.primary} text="加载算法详情..." />
        </View>
      </MainLayout>
    );
  }

  if (error || !algorithm) {
    return (
      <MainLayout title="算法详情" showBack>
        <View style={styles.stateContainer}>
          <EmptyState
            icon="error"
            title={error || '未找到算法'}
            description="请返回算法列表重试"
          />
        </View>
      </MainLayout>
    );
  }

  const statusInfo = ALGORITHM_STATUS_MAP[algorithm.status ?? 0] || ALGORITHM_STATUS_MAP[0];

  return (
    <MainLayout title="算法详情" showBack showBottomNav={false}>
      <View style={styles.container}>
        {/* 章节锚点导航条 */}
        <View style={styles.sectionNav}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.sectionNavContent}
          >
            {SECTIONS.map(section => {
              const isActive = activeSection === section.key;
              return (
                <TouchableOpacity
                  key={section.key}
                  style={[styles.sectionChip, isActive && styles.sectionChipActive]}
                  onPress={() => handleSectionPress(section.key)}
                  activeOpacity={0.7}
                >
                  <Ionicons
                    name={section.icon}
                    size={14}
                    color={isActive ? '#fff' : theme.colors.text.secondary}
                  />
                  <Text
                    style={[
                      styles.sectionChipText,
                      isActive && styles.sectionChipTextActive,
                    ]}
                  >
                    {section.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>

        <ScrollView
          ref={scrollRef}
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          onScroll={handleScroll}
          scrollEventThrottle={16}
        >
          {/* Hero 区域 */}
          <LinearGradient
            colors={['#3B82F6', '#6366F1']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.hero}
          >
            <View style={styles.heroTop}>
              <View style={styles.heroIconWrap}>
                <Ionicons name="flash" size={28} color="#fff" />
              </View>
              <View style={styles.heroBadges}>
                <View style={[styles.heroBadge, { backgroundColor: statusInfo.bgColor }]}>
                  <Text style={[styles.heroBadgeText, { color: statusInfo.color }]}>
                    {statusInfo.label}
                  </Text>
                </View>
                {algorithm.version && (
                  <View style={styles.heroBadge}>
                    <Text style={styles.heroBadgeTextLight}>v{algorithm.version}</Text>
                  </View>
                )}
              </View>
            </View>

            <Text style={styles.heroTitle} numberOfLines={2}>
              {algorithm.name}
            </Text>
            {algorithm.type && (
              <Text style={styles.heroSubtitle}>{algorithm.type}</Text>
            )}
            {algorithm.description && (
              <Text style={styles.heroDesc} numberOfLines={3}>
                {algorithm.description}
              </Text>
            )}

            {/* 关键指标摘要 */}
            <View style={styles.heroMetrics}>
              <View style={styles.heroMetricItem}>
                <Text style={styles.heroMetricValue}>
                  {monitor?.todayCallCount ?? '—'}
                </Text>
                <Text style={styles.heroMetricLabel}>今日调用</Text>
              </View>
              <View style={styles.heroMetricDivider} />
              <View style={styles.heroMetricItem}>
                <Text style={styles.heroMetricValue}>
                  {monitor?.avgTime ? `${monitor.avgTime.toFixed(0)}ms` : '—'}
                </Text>
                <Text style={styles.heroMetricLabel}>平均耗时</Text>
              </View>
              <View style={styles.heroMetricDivider} />
              <View style={styles.heroMetricItem}>
                <Text style={styles.heroMetricValue}>
                  {monitor?.successRate ? `${(monitor.successRate * 100).toFixed(1)}%` : '—'}
                </Text>
                <Text style={styles.heroMetricLabel}>成功率</Text>
              </View>
            </View>
          </LinearGradient>

          {/* 基本信息 */}
          <View
            onLayout={handleSectionLayout('basic')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="information-circle" title="基本信息" />
            <View style={styles.card}>
              <InfoRow label="算法名称" value={algorithm.name} />
              <InfoRow label="算法类型" value={algorithm.type} />
              <InfoRow label="当前版本" value={algorithm.version ? `v${algorithm.version}` : '—'} />
              <InfoRow label="状态" value={statusInfo.label} />
              <InfoRow label="模型大小" value={algorithm.size || '—'} />
              <InfoRow label="计算量" value={algorithm.flops || '—'} />
              <InfoRow label="导入路径" value={algorithm.importPath || algorithm.path || '—'} mono />
              <InfoRow label="创建时间" value={algorithm.createTime || '—'} />
            </View>
          </View>

          {/* 算法描述 */}
          <View
            onLayout={handleSectionLayout('description')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="document-text" title="算法描述" />
            <View style={styles.card}>
              {algorithm.description ? (
                <Text style={styles.descriptionText}>
                  {algorithm.description}
                </Text>
              ) : (
                <Text style={styles.emptyInlineText}>暂无算法描述</Text>
              )}
            </View>
          </View>

          {/* 效果样例 */}
          <View
            onLayout={handleSectionLayout('samples')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="images" title="效果样例" />
            <View style={styles.card}>
              <Text style={styles.sampleHint}>
                通过算法处理前后的对比图直观展示去雾效果
              </Text>
              <View style={styles.sampleCompareRow}>
                <View style={styles.sampleImageBox}>
                  <View style={styles.sampleImagePlaceholder}>
                    <Ionicons name="cloud" size={36} color={theme.colors.text.tertiary} />
                    <Text style={styles.sampleImageLabel}>雾霾原图</Text>
                  </View>
                </View>
                <View style={styles.sampleArrow}>
                  <Ionicons name="arrow-forward" size={20} color={theme.colors.primary} />
                </View>
                <View style={styles.sampleImageBox}>
                  <View style={[styles.sampleImagePlaceholder, styles.sampleImagePlaceholderClean]}>
                    <Ionicons name="sunny" size={36} color={theme.colors.status.success} />
                    <Text style={styles.sampleImageLabel}>去雾效果</Text>
                  </View>
                </View>
              </View>
              <Text style={styles.sampleNote}>
                实际效果取决于图像场景与雾霾浓度，可在「立即使用」中体验
              </Text>
            </View>
          </View>

          {/* 参数配置 */}
          <View
            onLayout={handleSectionLayout('params')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="options" title="参数配置" />
            <View style={styles.card}>
              {algorithm.params ? (
                <View style={styles.codeBlock}>
                  <Text style={styles.codeText}>{formatParams(algorithm.params)}</Text>
                </View>
              ) : (
                <Text style={styles.emptyInlineText}>该算法无可配置参数</Text>
              )}
            </View>
          </View>

          {/* 性能指标 */}
          <View
            onLayout={handleSectionLayout('metrics')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="analytics" title="性能指标" />
            <View style={styles.card}>
              {monitor ? (
                <View>
                  <MetricBar
                    label="调用总数"
                    value={monitor.callCount}
                    max={Math.max(monitor.callCount, 100)}
                    color="#3B82F6"
                    suffix="次"
                  />
                  <MetricBar
                    label="今日调用"
                    value={monitor.todayCallCount}
                    max={Math.max(monitor.todayCallCount, 50)}
                    color="#14B8A6"
                    suffix="次"
                  />
                  <MetricBar
                    label="平均耗时"
                    value={monitor.avgTime}
                    max={Math.max(monitor.avgTime, 1000)}
                    color="#F59E0B"
                    suffix="ms"
                  />
                  <MetricBar
                    label="成功率"
                    value={monitor.successRate * 100}
                    max={100}
                    color="#10B981"
                    suffix="%"
                    precision={1}
                  />
                </View>
              ) : (
                <Text style={styles.emptyInlineText}>暂无监控数据</Text>
              )}
            </View>
          </View>

          {/* 用户评价 */}
          <View
            onLayout={handleSectionLayout('reviews')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="star" title="用户评价" />
            <View style={styles.card}>
              <Text style={styles.emptyInlineText}>暂无用户评价</Text>
            </View>
          </View>

          {/* 相关链接 */}
          <View
            onLayout={handleSectionLayout('links')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="link" title="相关链接" />
            <View style={styles.card}>
              <Text style={styles.emptyInlineText}>暂无相关链接</Text>
            </View>
          </View>

          {/* 版本历史 */}
          <View
            onLayout={handleSectionLayout('versions')}
            style={styles.sectionWrap}
          >
            <SectionTitle icon="git-branch" title="版本历史" />
            <View style={styles.card}>
              {versions.length > 0 ? (
                <View style={styles.timeline}>
                  {versions.map((v, idx) => (
                    <View key={v.id} style={styles.timelineItem}>
                      <View style={styles.timelineLeft}>
                        <View
                          style={[
                            styles.timelineDot,
                            v.isActive && styles.timelineDotActive,
                          ]}
                        />
                        {idx < versions.length - 1 && <View style={styles.timelineLine} />}
                      </View>
                      <View style={styles.timelineContent}>
                        <View style={styles.timelineHeader}>
                          <Text style={styles.timelineVersion}>v{v.version}</Text>
                          {v.isActive && (
                            <Badge text="当前版本" variant="success" size="small" />
                          )}
                          {v.status === 0 && (
                            <Badge text="草稿" variant="secondary" size="small" />
                          )}
                        </View>
                        {v.changeLog && (
                          <Text style={styles.timelineChangeLog}>{v.changeLog}</Text>
                        )}
                        {v.createTime && (
                          <Text style={styles.timelineTime}>{v.createTime}</Text>
                        )}
                      </View>
                    </View>
                  ))}
                </View>
              ) : (
                <Text style={styles.emptyInlineText}>暂无版本历史</Text>
              )}
            </View>
          </View>

          {/* 底部留白给操作栏 */}
          <View style={{ height: 100 }} />
        </ScrollView>

        {/* 底部操作栏 */}
        <View style={styles.actionBar}>
          <TouchableOpacity
            style={styles.actionIconBtn}
            onPress={handleToggleFavorite}
            disabled={favoriteBusy}
            activeOpacity={0.7}
          >
            <Ionicons
              name={isFavorite ? 'heart' : 'heart-outline'}
              size={22}
              color={isFavorite ? theme.colors.status.error : theme.colors.text.secondary}
            />
            <Text
              style={[
                styles.actionIconText,
                isFavorite && styles.actionIconTextActive,
              ]}
            >
              {isFavorite ? '已收藏' : '收藏'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionIconBtn}
            onPress={handleShare}
            activeOpacity={0.7}
          >
            <Ionicons name="share-social-outline" size={22} color={theme.colors.text.secondary} />
            <Text style={styles.actionIconText}>分享</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionPrimaryBtn}
            onPress={handleUse}
            activeOpacity={0.85}
          >
            <LinearGradient
              colors={['#3B82F6', '#6366F1']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.actionPrimaryGradient}
            >
              <Ionicons name="rocket" size={18} color="#fff" />
              <Text style={styles.actionPrimaryText}>立即使用</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    </MainLayout>
  );
};

/** 章节标题 */
const SectionTitle: React.FC<{ icon: string; title: string }> = ({ icon, title }) => (
  <View style={styles.sectionTitleRow}>
    <View style={styles.sectionTitleIcon}>
      <Ionicons name={icon as any} size={16} color={theme.colors.primary} />
    </View>
    <Text style={styles.sectionTitleText}>{title}</Text>
  </View>
);

/** 信息行 */
const InfoRow: React.FC<{
  label: string;
  value: string;
  mono?: boolean;
}> = ({ label, value, mono }) => (
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

/** 指标条 */
const MetricBar: React.FC<{
  label: string;
  value: number;
  max: number;
  color: string;
  suffix?: string;
  precision?: number;
}> = ({ label, value, max, color, suffix = '', precision = 0 }) => {
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

/** 格式化参数 JSON */
function formatParams(params: string): string {
  try {
    return JSON.stringify(JSON.parse(params), null, 2);
  } catch {
    return params;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  stateContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: theme.colors.background.secondary,
  },
  // 章节锚点导航
  sectionNav: {
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
    maxHeight: 52,
  },
  sectionNavContent: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    gap: theme.spacing.sm,
  },
  sectionChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
    marginRight: 4,
  },
  sectionChipActive: {
    backgroundColor: theme.colors.primary,
  },
  sectionChipText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  sectionChipTextActive: {
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
  // 滚动容器
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: theme.spacing.xl,
  },
  // Hero
  hero: {
    padding: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    paddingBottom: theme.spacing.xl,
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.xxl,
    ...theme.layout.shadows.lg,
  },
  heroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.md,
  },
  heroIconWrap: {
    width: 56,
    height: 56,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  heroBadges: {
    flexDirection: 'row',
    gap: 6,
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
  },
  heroBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  heroBadgeText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.semibold,
  },
  heroBadgeTextLight: {
    fontSize: theme.typography.sizes.small,
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
  heroTitle: {
    fontSize: 26,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
    marginBottom: 6,
    letterSpacing: -0.5,
  },
  heroSubtitle: {
    fontSize: theme.typography.sizes.body,
    color: 'rgba(255,255,255,0.85)',
    marginBottom: theme.spacing.sm,
    fontWeight: theme.typography.weights.medium,
  },
  heroDesc: {
    fontSize: theme.typography.sizes.bodySmall,
    color: 'rgba(255,255,255,0.75)',
    lineHeight: 20,
    marginBottom: theme.spacing.md,
  },
  heroMetrics: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: theme.layout.borderRadius.lg,
    paddingVertical: theme.spacing.sm,
  },
  heroMetricItem: {
    flex: 1,
    alignItems: 'center',
  },
  heroMetricValue: {
    fontSize: 20,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  heroMetricLabel: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  heroMetricDivider: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.2)',
    marginVertical: 4,
  },
  // 章节
  sectionWrap: {
    marginTop: theme.spacing.lg,
    paddingHorizontal: theme.spacing.md,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  sectionTitleIcon: {
    width: 26,
    height: 26,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: `${theme.colors.primary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sectionTitleText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    letterSpacing: 0.3,
  },
  // 卡片
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  // 信息行
  infoRow: {
    flexDirection: 'row',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.colors.border.light,
  },
  infoLabel: {
    width: 80,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    fontWeight: theme.typography.weights.medium,
  },
  infoValue: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    fontWeight: theme.typography.weights.medium,
  },
  infoValueMono: {
    fontFamily: 'Menlo',
    fontSize: 12,
    color: theme.colors.text.secondary,
  },
  // 描述
  descriptionText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    lineHeight: 22,
  },
  emptyInlineText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: theme.spacing.md,
  },
  // 效果样例
  sampleHint: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginBottom: theme.spacing.md,
  },
  sampleCompareRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sampleImageBox: {
    flex: 1,
  },
  sampleImagePlaceholder: {
    aspectRatio: 1,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  sampleImagePlaceholderClean: {
    backgroundColor: '#ECFDF5',
    borderColor: '#A7F3D0',
  },
  sampleImageLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  sampleArrow: {
    paddingHorizontal: 2,
  },
  sampleNote: {
    fontSize: 11,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.sm,
    textAlign: 'center',
  },
  // 参数代码块
  codeBlock: {
    backgroundColor: '#0F172A',
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
  },
  codeText: {
    fontFamily: 'Menlo',
    fontSize: 12,
    color: '#E2E8F0',
    lineHeight: 18,
  },
  // 指标条
  metricBarWrap: {
    marginBottom: theme.spacing.md,
  },
  metricBarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  metricLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  metricValue: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
  },
  metricSuffix: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.regular,
    color: theme.colors.text.tertiary,
    marginLeft: 2,
  },
  metricBarTrack: {
    height: 8,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: 4,
    overflow: 'hidden',
  },
  metricBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  // 版本时间线
  timeline: {
    paddingTop: 4,
  },
  timelineItem: {
    flexDirection: 'row',
  },
  timelineLeft: {
    width: 20,
    alignItems: 'center',
  },
  timelineDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: theme.colors.text.tertiary,
    marginTop: 4,
  },
  timelineDotActive: {
    backgroundColor: theme.colors.primary,
    width: 12,
    height: 12,
    borderRadius: 6,
    marginTop: 3,
    borderWidth: 2,
    borderColor: '#fff',
    shadowColor: theme.colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 4,
    elevation: 4,
  },
  timelineLine: {
    flex: 1,
    width: 1,
    backgroundColor: theme.colors.border.light,
    marginTop: 4,
    marginBottom: 0,
    minHeight: 40,
  },
  timelineContent: {
    flex: 1,
    paddingBottom: theme.spacing.md,
    marginLeft: 8,
  },
  timelineHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 4,
  },
  timelineVersion: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  timelineChangeLog: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 18,
    marginBottom: 4,
  },
  timelineTime: {
    fontSize: 11,
    color: theme.colors.text.tertiary,
  },
  // 底部操作栏
  actionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: theme.colors.background.primary,
    borderTopWidth: 1,
    borderTopColor: theme.colors.border.light,
    ...theme.layout.shadows.md,
  },
  actionIconBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    minWidth: 52,
  },
  actionIconText: {
    fontSize: 11,
    color: theme.colors.text.secondary,
    marginTop: 2,
    fontWeight: theme.typography.weights.medium,
  },
  actionIconTextActive: {
    color: theme.colors.status.error,
  },
  actionPrimaryBtn: {
    flex: 1,
    borderRadius: theme.layout.borderRadius.md,
    overflow: 'hidden',
  },
  actionPrimaryGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 6,
  },
  actionPrimaryText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
});

export default AlgorithmScreen;
