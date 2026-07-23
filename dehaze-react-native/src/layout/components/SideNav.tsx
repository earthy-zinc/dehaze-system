/**
 * 侧边导航栏组件
 * 平板/桌面端侧边固定导航
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import type { IoniconName } from '@/components/Icon';
import {
  homeItem,
  menuSections,
  MenuItemData,
  MenuSection,
} from '../MenuConfig';
import type { RouteKeys } from '@/routes/types';

const SIDE_NAV_WIDTH = 260;

interface SideNavProps {
  currentRoute: RouteKeys;
  onNavigate: (route: RouteKeys) => void;
}

const SideNav: React.FC<SideNavProps> = ({ currentRoute, onNavigate }) => {
  const renderMenuItem = (item: MenuItemData, isActive: boolean) => (
    <TouchableOpacity
      key={item.route}
      style={[styles.menuItem, isActive && styles.activeMenuItem]}
      onPress={() => onNavigate(item.route)}
      activeOpacity={0.7}
    >
      <Ionicons
        name={item.icon as IoniconName}
        size={18}
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
            name={section.icon as IoniconName}
            size={12}
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

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.contentContainer}
      >
        {/* 首页 */}
        {renderMenuItem(homeItem, currentRoute === homeItem.route)}

        <View style={styles.divider} />

        {/* 分组菜单 */}
        {menuSections.map(renderSection)}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: SIDE_NAV_WIDTH,
    backgroundColor: colors.background.primary,
    borderRightWidth: 1,
    borderRightColor: colors.border.light,
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    paddingVertical: spacing.md,
  },
  divider: {
    height: 1,
    backgroundColor: colors.border.light,
    marginHorizontal: spacing.md,
    marginVertical: spacing.sm,
  },
  section: {
    marginBottom: spacing.xs,
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
    marginVertical: 1,
    borderRadius: layout.borderRadius.sm,
    gap: spacing.md,
  },
  activeMenuItem: {
    backgroundColor: colors.primaryLight,
  },
  menuItemText: {
    flex: 1,
    fontSize: 14,
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

export default SideNav;
