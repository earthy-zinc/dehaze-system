import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  RefreshControl,
  TouchableOpacity,
  Alert,
  Linking,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { MainLayout } from '@/layout';
import LoadingSpinner from '@/components/LoadingSpinner';
import EmptyState from '@/components/EmptyState';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import { taskApi } from './services/taskApi';
import type { TaskPage } from './services/taskApi';
import type { Task, TaskCategory, TaskStatus } from './types';
import {
  TASK_STATUS_MAP,
  TASK_TYPE_MAP,
  CATEGORY_FILTERS,
  TaskStatusEnum,
  isTerminal,
  isCancellable,
  formatTaskTime,
} from './types';

const PAGE_SIZE = 20;
/** 进行中任务轮询间隔 */
const POLL_INTERVAL = 3000;
/** 单任务最大轮询时长 */
const MAX_POLL_DURATION = 10 * 60 * 1000;

const TaskScreen: React.FC = () => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [statusFilter, setStatusFilter] = useState<TaskStatus | 'ALL'>('ALL');
  const [categoryFilter, setCategoryFilter] = useState<
    'ALL' | TaskCategory
  >('ALL');
  const [isLoading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [error, setError] = useState<string | null>(null);

  /** 进行中任务的轮询定时器 */
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** 轮询开始时间（防止永久轮询） */
  const pollStartRef = useRef<number>(0);

  const loadTasks = useCallback(
    async (page = 1, isRefresh = false) => {
      try {
        if (isRefresh) {
          setRefreshing(true);
        } else if (page === 1) {
          setLoading(true);
        }
        setError(null);

        const result: TaskPage = await taskApi.getPage({
          status: statusFilter === 'ALL' ? undefined : statusFilter,
          taskCategory: categoryFilter === 'ALL' ? undefined : categoryFilter,
          pageNum: page,
          pageSize: PAGE_SIZE,
        });

        const list = result.list || [];
        if (page === 1) {
          setTasks(list);
        } else {
          setTasks(prev => [...prev, ...list]);
        }
        setHasMore(list.length >= PAGE_SIZE);
        setCurrentPage(page);
      } catch (err: unknown) {
        const e = err as { msg?: string; message?: string };
        setError(e?.msg || e?.message || '加载任务列表失败');
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [statusFilter, categoryFilter],
  );

  useFocusEffect(
    useCallback(() => {
      loadTasks(1);
    }, [loadTasks]),
  );

  /** 有进行中任务时启动轮询 */
  useEffect(() => {
    const hasActive = tasks.some(t => !isTerminal(t));
    if (!hasActive) {
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    pollStartRef.current = Date.now();

    const poll = async () => {
      if (Date.now() - pollStartRef.current > MAX_POLL_DURATION) {
        if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        return;
      }
      try {
        // 轮询单个进行中任务的状态
        const activeTasks = tasks.filter(t => !isTerminal(t));
        const updated: Task[] = [];
        for (const t of activeTasks) {
          try {
            const fresh = await taskApi.getStatus(t.taskId);
            updated.push(fresh);
          } catch {
            // 忽略单个任务查询失败
          }
        }
        if (updated.length > 0) {
          setTasks(prev =>
            prev.map(t => {
              const fresh = updated.find(u => u.taskId === t.taskId);
              return fresh || t;
            }),
          );
        }
      } finally {
        pollTimerRef.current = setTimeout(poll, POLL_INTERVAL);
      }
    };

    pollTimerRef.current = setTimeout(poll, POLL_INTERVAL);
    return () => {
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [tasks]);

  const handleRefresh = useCallback(() => {
    loadTasks(1, true);
  }, [loadTasks]);

  const handleLoadMore = useCallback(() => {
    if (hasMore && !refreshing) {
      loadTasks(currentPage + 1);
    }
  }, [hasMore, refreshing, currentPage, loadTasks]);

  const handleStatusFilterChange = useCallback(
    (status: TaskStatus | 'ALL') => {
      setStatusFilter(status);
      setCurrentPage(1);
    },
    [],
  );

  const handleCategoryFilterChange = useCallback(
    (category: 'ALL' | TaskCategory) => {
      setCategoryFilter(category);
      setCurrentPage(1);
    },
    [],
  );

  const handleCancel = useCallback((task: Task) => {
    Alert.alert('取消任务', `确认取消任务 ${task.taskId} 吗？此操作不可恢复。`, [
      { text: '返回', style: 'cancel' },
      {
        text: '确认取消',
        style: 'destructive',
        onPress: async () => {
          try {
            await taskApi.cancel(task.taskId);
            setTasks(prev =>
              prev.map(t =>
                t.taskId === task.taskId
                  ? { ...t, status: TaskStatusEnum.CANCELLED }
                  : t,
              ),
            );
          } catch (err: unknown) {
            const e = err as { msg?: string; message?: string };
            Alert.alert(
              '取消失败',
              e?.msg || e?.message || '请稍后重试',
            );
          }
        },
      },
    ]);
  }, []);

  const handleDownload = useCallback(async (task: Task) => {
    if (task.status !== TaskStatusEnum.COMPLETED) {
      Alert.alert('提示', '任务尚未完成，无法下载');
      return;
    }
    if (task.downloadUrl) {
      // 优先使用任务返回的下载链接（通常是 OSS 签名 URL）
      try {
        await Linking.openURL(task.downloadUrl);
      } catch {
        Alert.alert('下载失败', '无法打开下载链接');
      }
      return;
    }
    // 回退到 SDK 下载接口
    try {
      await taskApi.download(task.taskId);
      Alert.alert('已触发下载', '请稍候查看下载目录');
    } catch (err: unknown) {
      const e = err as { msg?: string; message?: string };
      Alert.alert('下载失败', e?.msg || e?.message || '请稍后重试');
    }
  }, []);

  const renderItem = useCallback(
    ({ item }: { item: Task }) => {
      const statusInfo = TASK_STATUS_MAP[item.status] || TASK_STATUS_MAP[TaskStatusEnum.PENDING];
      const typeLabel = TASK_TYPE_MAP[item.taskType || ''] || item.taskType || '任务';
      const isActive = !isTerminal(item);
      const canCancel = isCancellable(item);
      const canDownload = item.status === TaskStatusEnum.COMPLETED;

      return (
        <View style={styles.taskCard}>
          {/* 头部：类型 + 状态 */}
          <View style={styles.cardHeader}>
            <View style={styles.typeRow}>
              <Icon name="task" size={16} color={theme.colors.secondary} />
              <Text style={styles.typeText}>{typeLabel}</Text>
            </View>
            <View
              style={[styles.statusBadge, { backgroundColor: statusInfo.bgColor }]}
            >
              <Text style={[styles.statusText, { color: statusInfo.color }]}>
                {statusInfo.label}
              </Text>
            </View>
          </View>

          {/* 任务 ID */}
          <Text style={styles.taskId} numberOfLines={1}>
            ID: {item.taskId}
          </Text>

          {/* 进度条 */}
          {(isActive || item.status === TaskStatusEnum.COMPLETED) && (
            <View style={styles.progressWrap}>
              <View style={styles.progressTrack}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: `${Math.min(100, Math.max(0, item.progress || 0))}%`,
                      backgroundColor: statusInfo.color,
                    },
                  ]}
                />
              </View>
              <Text style={styles.progressText}>{item.progress || 0}%</Text>
            </View>
          )}

          {/* 文件计数 */}
          {typeof item.processedFiles === 'number' &&
            typeof item.totalFiles === 'number' &&
            item.totalFiles > 0 && (
              <Text style={styles.fileCount}>
                已处理 {item.processedFiles}/{item.totalFiles}
              </Text>
            )}

          {/* 错误信息 */}
          {item.status === TaskStatusEnum.FAILED && item.error ? (
            <Text style={styles.errorText} numberOfLines={2}>
              错误：{item.error}
            </Text>
          ) : null}

          {/* 时间信息 */}
          <View style={styles.timeRow}>
            <Text style={styles.timeText}>
              创建：{formatTaskTime(item.createdAt)}
            </Text>
            {item.completedAt ? (
              <Text style={styles.timeText}>
                完成：{formatTaskTime(item.completedAt)}
              </Text>
            ) : null}
          </View>

          {/* 过期提示 */}
          {item.status === TaskStatusEnum.COMPLETED && item.expiresAt ? (
            <Text style={styles.expiresText}>
              过期时间：{formatTaskTime(item.expiresAt)}
            </Text>
          ) : null}

          {/* 操作按钮 */}
          <View style={styles.actionsRow}>
            {canCancel ? (
              <TouchableOpacity
                style={[styles.actionBtn, styles.cancelBtn]}
                onPress={() => handleCancel(item)}
                activeOpacity={0.8}
              >
                <Icon name="cancel" size={14} color={theme.colors.status.error} />
                <Text style={styles.cancelBtnText}>取消</Text>
              </TouchableOpacity>
            ) : null}
            {canDownload ? (
              <TouchableOpacity
                style={[styles.actionBtn, styles.downloadBtn]}
                onPress={() => handleDownload(item)}
                activeOpacity={0.8}
              >
                <Icon name="download" size={14} color={theme.colors.text.inverse} />
                <Text style={styles.downloadBtnText}>下载</Text>
              </TouchableOpacity>
            ) : null}
          </View>
        </View>
      );
    },
    [handleCancel, handleDownload],
  );

  const keyExtractor = useCallback((item: Task) => item.taskId, []);

  const statusFilters: { key: TaskStatus | 'ALL'; label: string }[] = [
    { key: 'ALL', label: '全部' },
    ...(Object.keys(TASK_STATUS_MAP).map(Number) as TaskStatus[]).map(key => ({
      key,
      label: TASK_STATUS_MAP[key].label,
    })),
  ];

  return (
    <MainLayout title="任务中心">
      <View style={styles.container}>
        {/* 任务类别筛选 */}
        <View style={styles.categoryBar}>
          {CATEGORY_FILTERS.map(f => {
            const isActive = categoryFilter === f.value;
            return (
              <TouchableOpacity
                key={f.value}
                style={[
                  styles.categoryChip,
                  isActive && styles.activeCategoryChip,
                ]}
                onPress={() => handleCategoryFilterChange(f.value)}
                activeOpacity={0.8}>
                <Text
                  style={[
                    styles.categoryChipText,
                    isActive && styles.activeCategoryChipText,
                  ]}>
                  {f.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* 状态筛选 */}
        <View style={styles.filterBar}>
          {statusFilters.map(f => {
            const isActive = statusFilter === f.key;
            return (
              <TouchableOpacity
                key={f.key}
                style={[
                  styles.filterChip,
                  isActive && styles.activeFilterChip,
                ]}
                onPress={() => handleStatusFilterChange(f.key)}
                activeOpacity={0.8}
              >
                <Text
                  style={[
                    styles.filterChipText,
                    isActive && styles.activeFilterChipText,
                  ]}
                >
                  {f.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* 任务列表 */}
        <FlatList
          data={tasks}
          renderItem={renderItem}
          keyExtractor={keyExtractor}
          contentContainerStyle={styles.listContainer}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={handleRefresh}
              tintColor={theme.colors.secondary}
              colors={[theme.colors.secondary]}
            />
          }
          onEndReached={handleLoadMore}
          onEndReachedThreshold={0.5}
          ListEmptyComponent={
            isLoading ? null : (
              <EmptyState
                icon="task"
                title="暂无任务"
                description={
                  error
                    ? error
                    : '还没有创建任何任务，可在数据集详情页导出数据集生成任务'
                }
              />
            )
          }
          ListFooterComponent={
            isLoading ? (
              <View style={styles.loadingContainer}>
                <LoadingSpinner size="large" color={theme.colors.secondary} />
              </View>
            ) : null
          }
        />
      </View>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  categoryBar: {
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.background.tertiary,
  },
  categoryChip: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: theme.colors.background.tertiary,
  },
  activeCategoryChip: {
    backgroundColor: theme.colors.secondary,
  },
  categoryChipText: {
    fontSize: 13,
    color: theme.colors.text.secondary,
    fontWeight: '500',
  },
  activeCategoryChipText: {
    color: theme.colors.text.inverse,
  },
  filterBar: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.background.tertiary,
  },
  filterChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: theme.colors.background.tertiary,
  },
  activeFilterChip: {
    backgroundColor: theme.colors.secondary,
  },
  filterChipText: {
    fontSize: 13,
    color: theme.colors.text.secondary,
    fontWeight: '500',
  },
  activeFilterChipText: {
    color: theme.colors.text.inverse,
  },
  listContainer: {
    padding: 16,
    paddingTop: 12,
  },
  taskCard: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: theme.colors.background.tertiary,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  typeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  typeText: {
    fontSize: 15,
    fontWeight: '600',
    color: theme.colors.text.primary,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
  },
  taskId: {
    fontSize: 12,
    color: theme.colors.text.tertiary,
    marginBottom: 8,
  },
  progressWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  progressTrack: {
    flex: 1,
    height: 6,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  progressText: {
    fontSize: 12,
    color: theme.colors.text.secondary,
    fontWeight: '500',
    minWidth: 36,
    textAlign: 'right',
  },
  fileCount: {
    fontSize: 12,
    color: theme.colors.text.secondary,
    marginBottom: 8,
  },
  errorText: {
    fontSize: 12,
    color: theme.colors.status.error,
    marginBottom: 8,
  },
  timeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  timeText: {
    fontSize: 11,
    color: theme.colors.text.tertiary,
  },
  expiresText: {
    fontSize: 11,
    color: theme.colors.status.warning,
    marginBottom: 8,
  },
  actionsRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 8,
    marginTop: 8,
    borderTopWidth: 1,
    borderTopColor: theme.colors.background.tertiary,
    paddingTop: 12,
  },
  actionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  cancelBtn: {
    backgroundColor: `${theme.colors.status.error}15`,
    borderWidth: 1,
    borderColor: `${theme.colors.status.error}40`,
  },
  cancelBtnText: {
    fontSize: 13,
    color: theme.colors.status.error,
    fontWeight: '500',
  },
  downloadBtn: {
    backgroundColor: theme.colors.secondary,
  },
  downloadBtnText: {
    fontSize: 13,
    color: theme.colors.text.inverse,
    fontWeight: '500',
  },
  loadingContainer: {
    paddingVertical: 20,
  },
});

export default TaskScreen;
