/**
 * 用户管理（管理侧）
 *
 * 列表/搜索/分页 + 新增/编辑/密码重置/状态管理
 * 权限：sys:user:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert,
  RefreshControl,
  Modal,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { UserAPI } from 'dehaze-sdk-js'
import type { UserPageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemUser'>;

const PAGE_SIZE = 15;

const SystemUserScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [list, setList] = useState<UserPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [keyword, setKeyword] = useState('');

  // 密码重置 Modal 状态
  const [passwordModalVisible, setPasswordModalVisible] = useState(false);
  const [passwordTarget, setPasswordTarget] = useState<UserPageVO | null>(null);
  const [newPassword, setNewPassword] = useState('');

  const fetchList = useCallback(async (pn: number, kw?: string) => {
    try {
      const res = await UserAPI.getPage({
        pageNum: pn,
        pageSize: PAGE_SIZE,
        keywords: kw || undefined,
      });
      const fetched = res?.list ?? [];
      const fetchedTotal = res?.total ?? 0;
      if (pn === 1) {
        setList(fetched);
      } else {
        setList((prev) => [...prev, ...fetched]);
      }
      setTotal(fetchedTotal);
      setPageNum(pn);
    } catch {
      Alert.alert('错误', '加载用户列表失败');
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchList(1, keyword).finally(() => setLoading(false));
  }, [fetchList, keyword]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchList(1, keyword);
    setRefreshing(false);
  }, [fetchList, keyword]);

  const handleLoadMore = useCallback(async () => {
    if (loadingMore || list.length >= total) return;
    setLoadingMore(true);
    await fetchList(pageNum + 1, keyword);
    setLoadingMore(false);
  }, [loadingMore, list.length, total, pageNum, fetchList, keyword]);

  const handleSearch = useCallback((text: string) => {
    setKeyword(text);
  }, []);

  const handleAdd = () => {
    if (!hasPerm('sys:user:add')) {
      Alert.alert('提示', '无新增权限');
      return;
    }
    navigation.navigate('SystemUserForm', {});
  };

  const handleEdit = (userId: number) => {
    if (!hasPerm('sys:user:edit')) {
      Alert.alert('提示', '无编辑权限');
      return;
    }
    navigation.navigate('SystemUserForm', { userId });
  };

  const handleDelete = (user: UserPageVO) => {
    if (!hasPerm('sys:user:delete')) {
      Alert.alert('提示', '无删除权限');
      return;
    }
    Alert.alert('确认删除', `确定要删除用户"${user.nickname}"吗？`, [
      { text: '取消', style: 'cancel' },
      {
        text: '确定',
        style: 'destructive',
        onPress: async () => {
          try {
            await UserAPI.deleteByIds(String(user.id));
            fetchList(1, keyword);
          } catch {
            Alert.alert('错误', '删除失败');
          }
        },
      },
    ]);
  };

  const handleResetPassword = (user: UserPageVO) => {
    if (!hasPerm('sys:user:edit')) {
      Alert.alert('提示', '无操作权限');
      return;
    }
    setPasswordTarget(user);
    setNewPassword('');
    setPasswordModalVisible(true);
  };

  const handleConfirmResetPassword = async () => {
    if (!newPassword || newPassword.length < 6) {
      Alert.alert('提示', '密码至少 6 位');
      return;
    }
    if (!passwordTarget) return;
    try {
      await UserAPI.updatePassword(passwordTarget.id!, newPassword);
      setPasswordModalVisible(false);
      Alert.alert('成功', '密码重置成功');
    } catch {
      Alert.alert('错误', '密码重置失败');
    }
  };

  const handleToggleStatus = (user: UserPageVO) => {
    if (!hasPerm('sys:user:edit')) {
      Alert.alert('提示', '无操作权限');
      return;
    }
    const newStatus = user.status === 1 ? 0 : 1;
    const action = newStatus === 0 ? '禁用' : '启用';
    Alert.alert(`确认${action}`, `确定要${action}用户"${user.nickname}"吗？`, [
      { text: '取消', style: 'cancel' },
      {
        text: '确定',
        onPress: async () => {
          try {
            await UserAPI.updateStatus(user.id!, newStatus);
            fetchList(1, keyword);
          } catch {
            Alert.alert('错误', '操作失败');
          }
        },
      },
    ]);
  };

  const renderItem = ({ item }: { item: UserPageVO }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.cardInfo}>
          <Text style={styles.cardName}>{item.nickname}</Text>
          <Text style={styles.cardUsername}>@{item.username}</Text>
        </View>
        <View style={[styles.statusBadge, item.status === 1 ? styles.statusEnabled : styles.statusDisabled]}>
          <Text style={[styles.statusText, item.status === 1 ? styles.statusTextEnabled : styles.statusTextDisabled]}>
            {item.status === 1 ? '启用' : '禁用'}
          </Text>
        </View>
      </View>
      {(item.deptName || item.roleNames) && (
        <View style={styles.cardMeta}>
          {item.deptName && <Text style={styles.metaText}>🏢 {item.deptName}</Text>}
          {item.roleNames && <Text style={styles.metaText}>🎭 {item.roleNames}</Text>}
        </View>
      )}
      <View style={styles.cardActions}>
        <TouchableOpacity style={styles.actionBtn} onPress={() => handleEdit(item.id!)}>
          <Ionicons name="create-outline" size={18} color={theme.colors.primary} />
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionBtn} onPress={() => handleResetPassword(item)}>
          <Ionicons name="key-outline" size={18} color={theme.colors.secondary} />
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionBtn} onPress={() => handleToggleStatus(item)}>
          <Ionicons
            name={item.status === 1 ? 'close-circle-outline' : 'checkmark-circle-outline'}
            size={18}
            color={item.status === 1 ? theme.colors.status.warning : theme.colors.status.success}
          />
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item)}>
          <Ionicons name="trash-outline" size={18} color={theme.colors.status.error} />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderFooter = () => {
    if (!loadingMore) return null;
    return (
      <View style={styles.footer}>
        <ActivityIndicator size="small" color={theme.colors.primary} />
      </View>
    );
  };

  const renderEmpty = () => {
    if (loading) return null;
    return (
      <View style={styles.empty}>
        <Ionicons name="people-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>{keyword ? '未找到匹配的用户' : '暂无用户数据'}</Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <AppHeader title="用户管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput
              style={styles.searchInput}
              placeholder="搜索用户名/昵称/手机号"
              placeholderTextColor={theme.colors.text.tertiary}
              value={keyword}
              onChangeText={handleSearch}
              returnKeyType="search"
            />
          </View>
          {hasPerm('sys:user:add') && (
            <TouchableOpacity style={styles.addBtn} onPress={handleAdd}>
              <Ionicons name="add" size={20} color="#fff" />
              <Text style={styles.addBtnText}>新增</Text>
            </TouchableOpacity>
          )}
        </View>
        <FlatList
          data={list}
          renderItem={renderItem}
          keyExtractor={(item) => String(item.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={handleRefresh}
              colors={[theme.colors.primary]}
              tintColor={theme.colors.primary}
            />
          }
          onEndReached={handleLoadMore}
          onEndReachedThreshold={0.3}
          ListFooterComponent={renderFooter}
          ListEmptyComponent={renderEmpty}
        />

        {/* 密码重置 Modal */}
        <Modal
          visible={passwordModalVisible}
          transparent
          animationType="fade"
          onRequestClose={() => setPasswordModalVisible(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>重置密码</Text>
              <Text style={styles.modalSubtitle}>
                为用户 {passwordTarget?.nickname || passwordTarget?.username} 设置新密码
              </Text>
              <TextInput
                style={styles.modalInput}
                placeholder="请输入新密码（至少6位）"
                placeholderTextColor={theme.colors.text.tertiary}
                secureTextEntry
                value={newPassword}
                onChangeText={setNewPassword}
                autoFocus
              />
              <View style={styles.modalActions}>
                <TouchableOpacity
                  style={styles.modalCancelBtn}
                  onPress={() => setPasswordModalVisible(false)}
                >
                  <Text style={styles.modalCancelText}>取消</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.modalConfirmBtn}
                  onPress={handleConfirmResetPassword}
                >
                  <Text style={styles.modalConfirmText}>确认重置</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    gap: theme.spacing.sm,
  },
  searchInputWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.sm,
    paddingHorizontal: theme.spacing.sm,
    height: 40,
    gap: 6,
  },
  searchInput: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    padding: 0,
  },
  addBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    gap: 4,
  },
  addBtnText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
  listContent: {
    paddingHorizontal: theme.spacing.md,
    paddingBottom: theme.spacing.xxxl,
  },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cardInfo: { flex: 1 },
  cardName: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  cardUsername: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 2,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: theme.layout.borderRadius.full,
  },
  statusEnabled: { backgroundColor: '#34d39920' },
  statusDisabled: { backgroundColor: '#ef444420' },
  statusText: {
    fontSize: theme.typography.sizes.tiny,
    fontWeight: theme.typography.weights.semibold,
  },
  statusTextEnabled: { color: '#34d399' },
  statusTextDisabled: { color: '#ef4444' },
  cardMeta: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.sm,
  },
  metaText: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.secondary,
  },
  cardActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.sm,
    paddingTop: theme.spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: theme.colors.border.light,
  },
  actionBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  footer: { paddingVertical: theme.spacing.md, alignItems: 'center' },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
  // 密码重置 Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '80%',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
    gap: theme.spacing.md,
  },
  modalTitle: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    textAlign: 'center',
  },
  modalInput: {
    borderWidth: 1,
    borderColor: theme.colors.border.light,
    borderRadius: theme.layout.borderRadius.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
  },
  modalActions: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    justifyContent: 'flex-end',
  },
  modalCancelBtn: {
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.background.tertiary,
  },
  modalCancelText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
  },
  modalConfirmBtn: {
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.primary,
  },
  modalConfirmText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
});

export default SystemUserScreen;
