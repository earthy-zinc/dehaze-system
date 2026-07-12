import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface IconProps {
  name: string;
  size?: number;
  color?: string;
  backgroundColor?: string;
  borderRadius?: number;
  style?: any;
}

// 简单的图标映射，使用文字代替Font Awesome图标
const iconMap: { [key: string]: string } = {
  'arrow-right': '→',
  'database': 'DB',
  'image': 'IMG',
  'brain': 'AI',
  'magic': '✨',
  'columns': '⧈',
  'layer-group': '⊞',
  'search-plus': '🔍',
  'sliders-h': '☰',
  'chart-line': '📊',
  'bolt': '⚡',
  'mobile-alt': '📱',
  'chart-bar': '📈',
  'check-circle': '✓',
  'arrow-down': '↓',
  'play': '▶',
  'pause': '⏸',
  'stop': '⏹',
  'refresh': '↻',
  'settings': '⚙',
  'user': '👤',
  'home': '🏠',
  'back': '←',
  'forward': '→',
  'up': '↑',
  'down': '↓',
  'clock': '🕒',
  'search': '🔍',
  'times': '×',
  'chevron-right': '›',
  'chevron-down': '⌄',
  'chevron-up': '⌃',
  'chevron-left': '‹',
  'download': '⬇',
  'upload': '⬆',
  'plus': '+',
  'minus': '−',
  'trash': '🗑',
  'edit': '✎',
  'info': 'ℹ',
  'warning': '⚠',
  'error': '✕',
  'success': '✓',
  'pending': '⏳',
  'cancel': '⊘',
  'file': '📄',
  'folder': '📁',
  'list': '☰',
  'grid': '▦',
  'eye': '👁',
  'tag': '🏷',
  'export': '📤',
  'task': '📋',
};

const Icon: React.FC<IconProps> = ({
  name,
  size = 24,
  color = '#3b82f6',
  backgroundColor,
  borderRadius = 12,
  style,
}) => {
  const iconSymbol = iconMap[name] || '?';

  if (backgroundColor) {
    return (
      <View
        style={[
          styles.container,
          {
            width: size * 1.5,
            height: size * 1.5,
            backgroundColor,
            borderRadius,
          },
          style,
        ]}
      >
        <Text
          style={[
            styles.icon,
            {
              fontSize: size * 0.7,
              color,
            },
          ]}
        >
          {iconSymbol}
        </Text>
      </View>
    );
  }

  return (
    <Text
      style={[
        styles.icon,
        {
          fontSize: size,
          color,
        },
        style,
      ]}
    >
      {iconSymbol}
    </Text>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    fontWeight: '600',
    textAlign: 'center',
    textAlignVertical: 'center',
  },
});

export default Icon;