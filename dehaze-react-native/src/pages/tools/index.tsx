/**
 * 工具 Tab (L1)
 *
 * 功能聚合中心：页内搜索 + 快捷入口横滑 + 功能网格（≤3 列）
 * 按 05-菜单与页面层级规划 2.2 节设计
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import type { ToolsStackParamList, TabParamList } from '@/routes/types';

interface QuickEntry {
  key: string;
  icon: string;
  label: string;
  route: keyof ToolsStackParamList | 'PersonalFavorites';
}

interface GridItem {
  key: string;
  icon: string;
  label: string;
  desc: string;
  route?: keyof ToolsStackParamList;
  comingSoon?: string;
}

const QUICK_ENTRIES: QuickEntry[] = [
  { key: 'history', icon: 'time-outline', label: '处理历史', route: 'Task' },
  { key: 'favorites', icon: 'heart-outline', label: '我的收藏', route: 'PersonalFavorites' },
  { key: 'batch', icon: 'layers-outline', label: '批量处理', route: 'Batch' },
  { key: 'algorithm', icon: 'git-network-outline', label: '算法选择', route: 'AlgorithmSelect' },
];

const GRID_ITEMS: GridItem[] = [
  { key: 'image-input', icon: 'images-outline', label: '图像输入', desc: '上传/拍照/样例库', route: 'ImageInput' },
  { key: 'algorithm-lib', icon: 'code-slash-outline', label: '算法库', desc: '浏览与对比算法', route: 'AlgorithmBrowse' },
  { key: 'dataset', icon: 'server-outline', label: '数据集', desc: '公开与共享数据集', route: 'Dataset' },
  { key: 'batch', icon: 'duplicate-outline', label: '批量处理', desc: '批量上传与执行', route: 'Batch' },
  { key: 'metrics', icon: 'analytics-outline', label: '指标管理', desc: 'PSNR/SSIM 查询', route: 'MetricsManage' },
  { key: 'api-doc', icon: 'document-text-outline', label: 'API 文档', desc: '开放接口文档', comingSoon: 'API 文档功能敬请期待' },
];

const GRID_NUM_COLUMNS = 3;

export default function ToolsScreen() {
  const navigation = useNavigation<NativeStackNavigationProp<ToolsStackParamList>>();
  const [searchText, setSearchText] = useState('');

  const navigateToToolRoute = useCallback(
    (route: keyof ToolsStackParamList) => {
      (navigation.navigate as (screen: string) => void)(route as string);
    },
    [navigation],
  );

  const handleQuickEntry = useCallback(
    (item: QuickEntry) => {
      if (item.route === 'PersonalFavorites') {
        navigation.getParent<BottomTabNavigationProp<TabParamList>>()?.navigate('Profile', { screen: 'PersonalFavorites' });
        return;
      }
      navigateToToolRoute(item.route);
    },
    [navigation, navigateToToolRoute],
  );

  const handleGridItem = useCallback(
    (item: GridItem) => {
      if (item.route) { navigateToToolRoute(item.route); }
      else if (item.comingSoon) { Alert.alert('提示', item.comingSoon); }
    },
    [navigateToToolRoute],
  );

  const renderGridItem = ({ item }: { item: GridItem }) => (
    <TouchableOpacity style={styles.gridItem} activeOpacity={0.7} onPress={() => handleGridItem(item)}>
      <View style={styles.gridIconWrap}>
        <Ionicons name={item.icon} size={24} color={colors.primary} />
      </View>
      <Text style={styles.gridLabel}>{item.label}</Text>
      <Text style={styles.gridDesc} numberOfLines={1}>{item.desc}</Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <Ionicons name="search-outline" size={18} color={colors.text.tertiary} style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="搜索算法、功能、文档..."
            placeholderTextColor={colors.text.tertiary}
            value={searchText}
            onChangeText={setSearchText}
            returnKeyType="search"
          />
          {searchText !== '' && (
            <TouchableOpacity onPress={() => setSearchText('')} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Ionicons name="close-circle" size={18} color={colors.text.tertiary} />
            </TouchableOpacity>
          )}
        </View>
        <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>快捷入口</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.quickRow}>
              {QUICK_ENTRIES.map(item => (
                <TouchableOpacity key={item.key} style={styles.quickItem} activeOpacity={0.7} onPress={() => handleQuickEntry(item)}>
                  <View style={styles.quickIconWrap}>
                    <Ionicons name={item.icon} size={22} color={colors.primary} />
                  </View>
                  <Text style={styles.quickLabel}>{item.label}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>全部功能</Text>
            <FlatList
              data={GRID_ITEMS}
              renderItem={renderGridItem}
              keyExtractor={item => item.key}
              numColumns={GRID_NUM_COLUMNS}
              columnWrapperStyle={styles.gridRow}
              scrollEnabled={false}
            />
          </View>
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background.secondary },
  container: { flex: 1 },
  searchBar: {
    flexDirection: 'row', alignItems: 'center', marginHorizontal: spacing.md, marginTop: spacing.sm,
    marginBottom: spacing.sm, height: 44, backgroundColor: colors.background.primary,
    borderRadius: layout.borderRadius.md, paddingHorizontal: spacing.md,
    borderWidth: 1, borderColor: colors.border.light,
  },
  searchIcon: { marginRight: spacing.sm },
  searchInput: { flex: 1, fontSize: 15, color: colors.text.primary, paddingVertical: 0 },
  scroll: { flex: 1 },
  scrollContent: { paddingBottom: spacing.xxxl },
  section: { marginTop: spacing.lg, paddingHorizontal: spacing.md },
  sectionTitle: { fontSize: 13, fontWeight: '600', color: colors.text.secondary, marginBottom: spacing.md },
  quickRow: { paddingRight: spacing.md, gap: spacing.md },
  quickItem: { alignItems: 'center', width: 72, gap: spacing.xs },
  quickIconWrap: { width: 48, height: 48, borderRadius: 16, backgroundColor: colors.primaryLight, justifyContent: 'center', alignItems: 'center' },
  quickLabel: { fontSize: 12, color: colors.text.secondary, textAlign: 'center' },
  gridRow: { gap: spacing.sm, marginBottom: spacing.sm },
  gridItem: {
    flex: 1, alignItems: 'center', paddingVertical: spacing.md, paddingHorizontal: spacing.xs,
    backgroundColor: colors.background.primary, borderRadius: layout.borderRadius.md, ...layout.shadows.sm,
  },
  gridIconWrap: { width: 44, height: 44, borderRadius: 12, backgroundColor: colors.primaryLight, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.sm },
  gridLabel: { fontSize: 13, fontWeight: '600', color: colors.text.primary, marginBottom: 2 },
  gridDesc: { fontSize: 11, color: colors.text.tertiary, textAlign: 'center' },
});
