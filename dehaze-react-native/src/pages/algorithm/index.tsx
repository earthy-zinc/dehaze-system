/**
 * 算法详情页面 (F-M03-004)
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
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Alert,
  Share,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import LinearGradient from 'react-native-linear-gradient';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { DehazeStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import LoadingSpinner from '@/components/LoadingSpinner';
import EmptyState from '@/components/EmptyState';
import Badge from '@/components/Badge';
import { useSectionScroll, type SectionDef } from '@/hooks/useSectionScroll';

import { AlgorithmAPI, FavoriteAPI } from 'dehaze-sdk-js';
import type {
  Algorithm,
  AlgorithmVersionVO,
  AlgorithmMonitorVO,
} from 'dehaze-sdk-js';

import SectionTitle from './components/SectionTitle';
import InfoRow from './components/InfoRow';
import MetricBar from './components/MetricBar';
import { styles } from './styles';

type Props = NativeStackScreenProps<DehazeStackParamList, 'Algorithm'>;

/** 算法状态枚举映射 (来自后端 AlgorithmStatusEnum) */
const ALGORITHM_STATUS_MAP: Record<
  number,
  { label: string; color: string; bgColor: string }
> = {
  1: {
    label: '草稿',
    color: theme.colors.text.tertiary,
    bgColor: theme.colors.background.tertiary,
  },
  2: {
    label: '测试中',
    color: theme.colors.badge.warning.text,
    bgColor: theme.colors.badge.warning.bg,
  },
  3: {
    label: '待审核',
    color: theme.colors.badge.info.text,
    bgColor: theme.colors.badge.info.bg,
  },
  4: {
    label: '已发布',
    color: theme.colors.badge.success.text,
    bgColor: theme.colors.badge.success.bg,
  },
  5: {
    label: '已停用',
    color: theme.colors.badge.error.text,
    bgColor: theme.colors.badge.error.bg,
  },
  6: {
    label: '已归档',
    color: theme.colors.text.secondary,
    bgColor: theme.colors.background.tertiary,
  },
};

/** 章节定义 */
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

/** 格式化参数 JSON */
function formatParams(params: string): string {
  try {
    return JSON.stringify(JSON.parse(params), null, 2);
  } catch {
    return params;
  }
}

const AlgorithmScreen: React.FC<Props> = ({ route, navigation }) => {
  const algorithmId = route.params?.algorithmId;

  // 数据状态
  const [algorithm, setAlgorithm] = useState<Algorithm | null>(null);
  const [monitor, setMonitor] = useState<AlgorithmMonitorVO | null>(null);
  const [versions, setVersions] = useState<AlgorithmVersionVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // UI 状态
  const [isFavorite, setIsFavorite] = useState(false);
  const [favoriteBusy, setFavoriteBusy] = useState(false);

  // 章节锚点滚动
  const {
    scrollRef,
    activeSection,
    handleSectionPress,
    handleSectionLayout,
    handleScroll,
  } = useSectionScroll(SECTIONS);

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
      const [monRes, verRes] = await Promise.allSettled([
        AlgorithmAPI.getMonitorData(algorithmId),
        AlgorithmAPI.getVersions(algorithmId),
      ]);
      if (monRes.status === 'fulfilled') setMonitor(monRes.value);
      if (verRes.status === 'fulfilled') setVersions(verRes.value || []);
    } catch (e: unknown) {
      const err = e as { message?: string };
      setError(err?.message || '加载算法详情失败');
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
    FavoriteAPI.getStatus('algorithm', algorithmId)
      .then(status => {
        setIsFavorite(status.favorited);
      })
      .catch(() => {
        /* 收藏服务不可用时静默忽略 */
      });
  }, [algorithmId]);

  /** 立即使用 → 跳转处理页 */
  const handleUse = useCallback(() => {
    if (!algorithm) return;
    navigation.navigate('Processing', { algorithmId: algorithm.id });
  }, [algorithm, navigation]);

  /** 切换收藏 */
  const handleToggleFavorite = useCallback(() => {
    if (!algorithm) return;
    setFavoriteBusy(true);
    const toggle = async () => {
      if (isFavorite) {
        // 已收藏：按收藏记录 ID 取消（查询目标算法对应的收藏记录）
        const page = await FavoriteAPI.getPage({
          targetType: 'algorithm',
          pageNum: 1,
          pageSize: 100,
        });
        const fav = (page?.list || []).find(f => f.targetId === algorithm.id);
        if (fav) {
          await FavoriteAPI.deleteByIds([fav.id]);
        }
      } else {
        await FavoriteAPI.add({ targetType: 'algorithm', targetId: algorithm.id });
      }
    };
    toggle()
      .then(() => setIsFavorite(!isFavorite))
      .catch(err => {
        Alert.alert('操作失败', err?.message || '收藏服务不可用，请稍后再试');
      })
      .finally(() => setFavoriteBusy(false));
  }, [algorithm, isFavorite]);

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
      <View style={styles.screenContainer}>
        <AppHeader title="算法详情" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.stateContainer}>
          <LoadingSpinner size="large" color={theme.colors.primary} text="加载算法详情..." />
        </View>
      </View>
    );
  }

  if (error || !algorithm) {
    return (
      <View style={styles.screenContainer}>
        <AppHeader title="算法详情" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.stateContainer}>
          <EmptyState
            icon="error"
            title={error || '未找到算法'}
            description="请返回算法列表重试"
          />
        </View>
      </View>
    );
  }

  const statusInfo = ALGORITHM_STATUS_MAP[algorithm.status ?? 0] || ALGORITHM_STATUS_MAP[0];

  return (
    <View style={styles.screenContainer}>
      <AppHeader title="算法详情" showBack onBackPress={() => navigation.goBack()} />
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
            colors={theme.colors.gradient.primary}
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
                    color={theme.colors.primary}
                    suffix="次"
                  />
                  <MetricBar
                    label="今日调用"
                    value={monitor.todayCallCount}
                    max={Math.max(monitor.todayCallCount, 50)}
                    color={theme.colors.secondary}
                    suffix="次"
                  />
                  <MetricBar
                    label="平均耗时"
                    value={monitor.avgTime}
                    max={Math.max(monitor.avgTime, 1000)}
                    color={theme.colors.status.warning}
                    suffix="ms"
                  />
                  <MetricBar
                    label="成功率"
                    value={monitor.successRate * 100}
                    max={100}
                    color={theme.colors.status.success}
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
          <View style={styles.bottomSpacer} />
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
              colors={theme.colors.gradient.primary}
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
    </View>
  );
};

export default AlgorithmScreen;