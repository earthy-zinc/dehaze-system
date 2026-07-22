/**
 * 底部导航栏组件
 * 移动端底部标签导航，支持图标和文字标签
 */
import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import { bottomTabs, RouteNames } from '../MenuConfig';

// 图标映射 - Ionicons 图标名称
const iconMap: { [key: string]: { default: string; active: string } } = {
  home: { default: 'home-outline', active: 'home' },
  image: { default: 'image-outline', active: 'image' },
  bulb: { default: 'bulb-outline', active: 'bulb' },
  cog: { default: 'cog-outline', active: 'cog' },
  albums: { default: 'albums-outline', active: 'albums' },
};

interface BottomTabBarProps {
  currentRoute: RouteNames;
  onTabPress: (route: RouteNames) => void;
}

const BottomTabBar: React.FC<BottomTabBarProps> = ({
  currentRoute,
  onTabPress,
}) => {
  const insets = useSafeAreaInsets();

  return (
    <View
      style={[
        styles.container,
        { paddingBottom: Math.max(insets.bottom, spacing.sm) },
      ]}
    >
      {bottomTabs.map(tab => {
        const isActive = currentRoute === tab.route;
        const icons = iconMap[tab.icon] || { default: 'ellipse-outline', active: 'ellipse' };
        const iconName = isActive ? icons.active : icons.default;

        return (
          <TouchableOpacity
            key={tab.route}
            style={styles.tabItem}
            onPress={() => onTabPress(tab.route)}
            activeOpacity={0.7}
          >
            <View
              style={[styles.iconContainer, isActive && styles.activeIconContainer]}
            >
              <Ionicons
                name={iconName}
                size={22}
                color={isActive ? colors.primary : colors.text.secondary}
              />
            </View>
            <Text
              style={[styles.tabLabel, isActive && styles.activeTabLabel]}
              numberOfLines={1}
            >
              {tab.title}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: colors.background.primary,
    borderTopWidth: 1,
    borderTopColor: colors.border.light,
    paddingTop: spacing.sm,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -2 },
        shadowOpacity: 0.05,
        shadowRadius: 8,
      },
      android: {
        elevation: 8,
      },
    }),
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xs,
  },
  iconContainer: {
    width: 40,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 2,
  },
  activeIconContainer: {
    backgroundColor: colors.primaryLight,
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: '500',
    color: colors.text.secondary,
    marginTop: 2,
  },
  activeTabLabel: {
    color: colors.primary,
    fontWeight: '600',
  },
});

export default BottomTabBar;
