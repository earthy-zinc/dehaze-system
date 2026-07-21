import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';

interface IconProps {
  name: string;
  size?: number;
  color?: string;
  backgroundColor?: string;
  borderRadius?: number;
  style?: ViewStyle;
}

// 业务自定义图标名 -> Ionicons 名称映射
// 统一使用 Ionicons 字体，避免文字符号/emoji 在不同设备渲染不一致
const iconMap: { [key: string]: string } = {
  // 方向
  'arrow-right': 'arrow-forward',
  'arrow-down': 'arrow-down',
  'chevron-right': 'chevron-forward',
  'chevron-down': 'chevron-down',
  'chevron-up': 'chevron-up',
  'chevron-forward': 'chevron-forward',
  'back': 'arrow-back',
  'forward': 'arrow-forward',
  'up': 'arrow-up',
  'down': 'arrow-down',
  // 功能
  'database': 'server',
  'image': 'image-outline',
  'brain': 'bulb-outline',
  'magic': 'sparkles',
  'columns': 'grid-outline',
  'layer-group': 'layers-outline',
  'search-plus': 'search',
  'sliders-h': 'options-outline',
  'chart-line': 'analytics-outline',
  'bolt': 'flash',
  'mobile-alt': 'phone-portrait',
  'chart-bar': 'bar-chart-outline',
  'check-circle': 'checkmark-circle',
  'play': 'play',
  'pause': 'pause',
  'stop': 'stop',
  'refresh': 'refresh',
  'settings': 'settings',
  'user': 'person',
  'home': 'home',
  'clock': 'time',
  'search': 'search',
  'times': 'close',
  'download': 'download',
  'upload': 'cloud-upload',
  'plus': 'add',
  'minus': 'remove',
  'trash': 'trash',
  'edit': 'create',
  'info': 'information-circle',
  'warning': 'warning',
  'error': 'alert-circle',
  'success': 'checkmark-circle',
  'pending': 'hourglass',
  'cancel': 'close-circle',
  'file': 'document',
  'folder': 'folder',
  'folder-open': 'folder-open',
  'folder-open-outline': 'folder-open-outline',
  'list': 'list',
  'grid': 'grid',
  'eye': 'eye',
  'tag': 'pricetag',
  'export': 'share',
  'task': 'clipboard',
  // 底部 Tab
  'tab-home': 'home',
  'tab-image': 'images',
  'tab-algorithm': 'bulb',
  'tab-dataset': 'server',
  'tab-task': 'clipboard',
};

const Icon: React.FC<IconProps> = ({
  name,
  size = 24,
  color = '#3b82f6',
  backgroundColor,
  borderRadius = 12,
  style,
}) => {
  const ioniconName = iconMap[name] || name;

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
        <Ionicons name={ioniconName} size={size} color={color} />
      </View>
    );
  }

  return <Ionicons name={ioniconName} size={size} color={color} style={style} />;
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default Icon;
