/**
 * 顶部导航栏组件
 *
 * L1（Tab 首页）：isHome 显示品牌 Logo + 标题，其他 Tab 只显示标题居左
 * L2（Stack 子页面）：showBack 返回按钮 + 居中标题 + rightActions 操作 slot
 */
import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import LinearGradient from 'react-native-linear-gradient';

interface AppHeaderProps {
  title?: string;
  showBack?: boolean;
  isHome?: boolean;
  onBackPress?: () => void;
  rightActions?: React.ReactNode;
}

const AppHeader: React.FC<AppHeaderProps> = ({
  title = '图像去雾系统',
  showBack = false,
  isHome = false,
  onBackPress,
  rightActions,
}) => {
  const insets = useSafeAreaInsets();

  return (
    <View style={[styles.wrapper, { paddingTop: insets.top }]}>
      <StatusBar
        barStyle="dark-content"
        backgroundColor="transparent"
        translucent
      />
      <View style={styles.content}>
        {/* 左侧区域 */}
        <View style={styles.leftSection}>
          {showBack ? (
            <TouchableOpacity
              style={styles.iconButton}
              onPress={onBackPress}
              activeOpacity={0.7}
            >
              <Ionicons
                name="chevron-back"
                size={24}
                color={colors.text.primary}
              />
            </TouchableOpacity>
          ) : isHome ? (
            <View style={styles.logoContainer}>
              <LinearGradient
                colors={[colors.primary, '#6366f1']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.logoGradient}
              >
                <Ionicons name="cloud-outline" size={18} color="#fff" />
              </LinearGradient>
              <Text style={styles.title}>{title}</Text>
            </View>
          ) : (
            <Text style={styles.title}>{title}</Text>
          )}
        </View>

        {/* 居中标题（L2 有返回按钮时） */}
        {showBack && (
          <View style={styles.centerSection}>
            <Text style={styles.centerTitle} numberOfLines={1}>
              {title}
            </Text>
          </View>
        )}

        {/* 右侧区域 */}
        <View style={styles.rightSection}>
          {rightActions}
          {showBack && <View style={styles.iconButton} />}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: {
    backgroundColor: colors.background.primary,
    ...layout.shadows.sm,
    zIndex: 100,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    height: 56,
  },
  leftSection: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  centerSection: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
    pointerEvents: 'none',
  },
  centerTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.text.primary,
    maxWidth: '60%',
  },
  rightSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  logoGradient: {
    width: 32,
    height: 32,
    borderRadius: layout.borderRadius.sm,
    justifyContent: 'center',
    alignItems: 'center',
    ...layout.shadows.sm,
  },
  title: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.text.primary,
  },
  iconButton: {
    width: 40,
    height: 40,
    borderRadius: layout.borderRadius.sm,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default AppHeader;
