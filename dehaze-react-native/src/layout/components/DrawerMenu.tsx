/**
 * 侧边抽屉菜单组件
 * 支持手势滑动打开，提供应用全局导航选项
 */
import React, { useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  ScrollView,
  TouchableWithoutFeedback,
  Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Ionicons from 'react-native-vector-icons/Ionicons';
import LinearGradient from 'react-native-linear-gradient';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import {
  homeItem,
  menuSections,
  RouteNames,
  MenuItemData,
  MenuSection,
} from '../MenuConfig';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const DRAWER_WIDTH = Math.min(320, SCREEN_WIDTH * 0.85);

interface DrawerMenuProps {
  visible: boolean;
  currentRoute: RouteNames;
  onClose: () => void;
  onNavigate: (route: RouteNames) => void;
}

const DrawerMenu: React.FC<DrawerMenuProps> = ({
  visible,
  currentRoute,
  onClose,
  onNavigate,
}) => {
  const insets = useSafeAreaInsets();
  const translateX = useRef(new Animated.Value(DRAWER_WIDTH)).current;
  const overlayOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(translateX, {
          toValue: 0,
          useNativeDriver: true,
          tension: 65,
          friction: 11,
        }),
        Animated.timing(overlayOpacity, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();
    } else {
      Animated.parallel([
        Animated.spring(translateX, {
          toValue: DRAWER_WIDTH,
          useNativeDriver: true,
          tension: 65,
          friction: 11,
        }),
        Animated.timing(overlayOpacity, {
          toValue: 0,
          duration: 150,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [visible, translateX, overlayOpacity]);

  const handleItemPress = useCallback(
    (route: RouteNames) => {
      onNavigate(route);
      onClose();
    },
    [onNavigate, onClose],
  );

  const renderMenuItem = (item: MenuItemData, isActive: boolean) => (
    <TouchableOpacity
      key={item.route}
      style={[styles.menuItem, isActive && styles.activeMenuItem]}
      onPress={() => handleItemPress(item.route)}
      activeOpacity={0.7}
    >
      <Ionicons
        name={item.icon as any}
        size={20}
        color={isActive ? colors.primary : colors.text.secondary}
      />
      <Text style={[styles.menuItemText, isActive && styles.activeMenuItemText]}>
        {item.title}
      </Text>
      {item.badge && (
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{item.badge}</Text>
        </View>
      )}
    </TouchableOpacity>
  );

  const renderSection = (section: MenuSection) => (
    <View key={section.title} style={styles.section}>
      <View style={styles.sectionHeader}>
        {section.icon && (
          <Ionicons
            name={section.icon as any}
            size={14}
            color={colors.text.muted}
            style={styles.sectionIcon}
          />
        )}
        <Text style={styles.sectionTitle}>{section.title}</Text>
      </View>
      {section.items.map(item =>
        renderMenuItem(item, currentRoute === item.route),
      )}
    </View>
  );

  if (!visible) return null;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
      {/* 遮罩层 */}
      <TouchableWithoutFeedback onPress={onClose}>
        <Animated.View
          style={[
            styles.overlay,
            {
              opacity: overlayOpacity,
            },
          ]}
        />
      </TouchableWithoutFeedback>

      {/* 抽屉面板 */}
      <Animated.View
        style={[
          styles.drawer,
          {
            transform: [{ translateX }],
            paddingTop: insets.top,
            paddingBottom: insets.bottom,
          },
        ]}
      >
        {/* 头部 */}
        <LinearGradient
          colors={[colors.primary, '#6366f1']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.header}
        >
          <View style={styles.headerContent}>
            <View style={styles.headerLeft}>
              <View style={styles.logoWrapper}>
                <Ionicons name="cloud-outline" size={28} color="#fff" />
              </View>
              <Text style={styles.headerTitle}>图像去雾系统</Text>
            </View>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={onClose}
              activeOpacity={0.7}
            >
              <Ionicons name="close" size={24} color="rgba(255,255,255,0.8)" />
            </TouchableOpacity>
          </View>
          <Text style={styles.headerSubtitle}>功能菜单</Text>
        </LinearGradient>

        {/* 菜单内容 */}
        <ScrollView
          style={styles.menuContent}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.menuContentContainer}
        >
          {/* 首页 */}
          {renderMenuItem(homeItem, currentRoute === homeItem.route)}

          <View style={styles.divider} />

          {/* 分组菜单 */}
          {menuSections.map(renderSection)}
        </ScrollView>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  drawer: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    width: DRAWER_WIDTH,
    backgroundColor: colors.background.primary,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: -4, height: 0 },
        shadowOpacity: 0.15,
        shadowRadius: 20,
      },
      android: {
        elevation: 16,
      },
    }),
  },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.md,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  logoWrapper: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: spacing.sm,
  },
  closeButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  menuContent: {
    flex: 1,
  },
  menuContentContainer: {
    paddingVertical: spacing.md,
  },
  divider: {
    height: 1,
    backgroundColor: colors.border.light,
    marginHorizontal: spacing.md,
    marginVertical: spacing.sm,
  },
  section: {
    marginBottom: spacing.sm,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  sectionIcon: {
    marginRight: spacing.xs,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    marginHorizontal: spacing.sm,
    borderRadius: layout.borderRadius.sm,
    gap: spacing.md,
  },
  activeMenuItem: {
    backgroundColor: colors.primaryLight,
  },
  menuItemText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '400',
    color: colors.text.primary,
  },
  activeMenuItemText: {
    color: colors.primary,
    fontWeight: '600',
  },
  badge: {
    backgroundColor: colors.primary,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
  },
  badgeText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
  },
});

export default DrawerMenu;
