/**
 * 顶部导航栏组件
 * 支持标题显示、返回按钮、操作按钮和菜单按钮
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
  showMenu?: boolean;
  showSearch?: boolean;
  onBackPress?: () => void;
  onMenuPress?: () => void;
  onSearchPress?: () => void;
  rightActions?: React.ReactNode;
  transparent?: boolean;
}

const AppHeader: React.FC<AppHeaderProps> = ({
  title = '图像去雾系统',
  showBack = false,
  showMenu = true,
  showSearch = true,
  onBackPress,
  onMenuPress,
  onSearchPress,
  rightActions,
  transparent = false,
}) => {
  const insets = useSafeAreaInsets();

  const renderContent = () => (
    <View style={[styles.container, { paddingTop: insets.top }]}>
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
          ) : (
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
          )}
        </View>

        {/* 右侧区域 */}
        <View style={styles.rightSection}>
          {rightActions}
          {showSearch && (
            <TouchableOpacity
              style={styles.iconButton}
              onPress={onSearchPress}
              activeOpacity={0.7}
            >
              <Ionicons name="search" size={22} color={colors.text.primary} />
            </TouchableOpacity>
          )}
          {showMenu && (
            <TouchableOpacity
              style={styles.iconButton}
              onPress={onMenuPress}
              activeOpacity={0.7}
            >
              <Ionicons name="menu" size={24} color={colors.text.primary} />
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );

  if (transparent) {
    return renderContent();
  }

  return (
    <View style={styles.wrapper}>
      {renderContent()}
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: {
    backgroundColor: colors.background.primary,
    ...layout.shadows.sm,
    zIndex: 100,
  },
  container: {
    backgroundColor: 'transparent',
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
    fontSize: 16,
    fontWeight: '700',
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
