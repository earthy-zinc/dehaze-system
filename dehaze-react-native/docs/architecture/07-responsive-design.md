# 响应式设计策略

**文档版本**: v1.0
**最后更新**: 2025-11-22
**项目名称**: dehaze-react-native
**目标平台**: iOS、Android

---

## 📋 文档概述

本文档详细描述了Dehaze React Native应用的响应式设计策略，包括设备适配方案、布局系统、尺寸规范、字体系统、图片处理和用户体验优化。基于React Native的跨平台特性，提供在不同屏幕尺寸和设备类型上的一致性用户体验。

---

## 🎯 响应式设计原则

### 移动端优先设计

#### 1. 设备兼容性
- **多平台支持**: iPhone、iPad、Android手机、Android平板
- **屏幕密度适配**: 支持@1x、@2x、@3x等多种像素密度
- **屏幕方向适配**: 支持横屏和竖屏切换
- **安全区域处理**: 适配刘海屏、虚拟导航栏等特殊区域

#### 2. 用户体验一致性
- **触摸友好**: 保证最小44px的触控热区
- **布局适配**: 根据屏幕尺寸调整布局结构
- **内容可读性**: 字体大小和行距自适应调整
- **交互效率**: 合理的手势操作和导航设计

#### 3. 性能优化
- **按需加载**: 根据设备性能调整加载策略
- **内存管理**: 大屏幕设备合理控制内存使用
- **图片优化**: 根据屏幕密度提供合适尺寸的图片
- **动画性能**: 确保不同设备上的动画流畅性

---

## 📱 设备分类与断点

### 设备分类体系

```mermaid
graph LR
    subgraph "设备类型"
        PHONE[手机设备<br/>4.7" - 7"]
        TABLET[平板设备<br/>8" - 13"]
        FOLDABLE[折叠设备<br/>可变屏幕]
    end

    subgraph "屏幕方向"
        PORTRAIT[竖屏<br/>高度 > 宽度]
        LANDSCAPE[横屏<br/>宽度 > 高度]
    end

    subgraph "像素密度"
        LDPI[@1x 低密度<br/>~120 DPI]
        MDPI[@2x 中密度<br/>~160 DPI]
        HDPI[@2x 高密度<br/>~240 DPI]
        XHDPI[@2x 超高密度<br/>~320 DPI]
        XXHDPI[@3x 超超高密度<br/>~480 DPI]
    end

    PHONE --> PORTRAIT
    PHONE --> LANDSCAPE
    TABLET --> PORTRAIT
    TABLET --> LANDSCAPE
    FOLDABLE --> PORTRAIT
    FOLDABLE --> LANDSCAPE
```

### 响应式断点定义

```typescript
// 设备断点配置
const Breakpoints = {
  // 屏幕宽度断点
  SCREEN_WIDTH: {
    SMALL: 375,    // iPhone SE
    MEDIUM: 414,   // iPhone 12 Pro
    LARGE: 768,    // iPad mini
    EXTRA_LARGE: 1024, // iPad
  },

  // 屏幕高度断点
  SCREEN_HEIGHT: {
    SHORT: 667,    // iPhone SE
    MEDIUM: 812,   // iPhone 12 Pro
    TALL: 1024,    // iPad mini
    EXTRA_TALL: 1366, // iPad
  },

  // 像素密度断点
  PIXEL_DENSITY: {
    LOW: 1,        // @1x
    MEDIUM: 2,     // @2x
    HIGH: 3,       // @3x
  }
};

// 设备类型枚举
enum DeviceType {
  PHONE_SMALL = 'phone_small',
  PHONE_MEDIUM = 'phone_medium',
  TABLET_SMALL = 'tablet_small',
  TABLET_MEDIUM = 'tablet_medium',
  TABLET_LARGE = 'tablet_large',
}

// 屏幕方向枚举
enum Orientation {
  PORTRAIT = 'portrait',
  LANDSCAPE = 'landscape',
}
```

### 设备检测工具

```typescript
// 响应式工具Hook
const useResponsive = () => {
  const [dimensions, setDimensions] = useState(() => {
    const { width, height, scale } = Dimensions.get('window');
    return { width, height, scale };
  });

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions({
        width: window.width,
        height: window.height,
        scale: window.scale || PixelRatio.get(),
      });
    });

    return () => subscription?.remove();
  }, []);

  // 设备类型检测
  const deviceType = useMemo(() => {
    const { width, height } = dimensions;

    if (width < Breakpoints.SCREEN_WIDTH.MEDIUM) {
      return DeviceType.PHONE_SMALL;
    } else if (width < Breakpoints.SCREEN_WIDTH.LARGE) {
      return DeviceType.PHONE_MEDIUM;
    } else if (width < Breakpoints.SCREEN_WIDTH.EXTRA_LARGE) {
      return DeviceType.TABLET_SMALL;
    } else {
      return DeviceType.TABLET_MEDIUM;
    }
  }, [dimensions]);

  // 屏幕方向检测
  const orientation = useMemo(() => {
    const { width, height } = dimensions;
    return width > height ? Orientation.LANDSCAPE : Orientation.PORTRAIT;
  }, [dimensions]);

  // 是否为平板设备
  const isTablet = useMemo(() => {
    return deviceType.includes('tablet');
  }, [deviceType]);

  // 是否为手机设备
  const isPhone = useMemo(() => {
    return deviceType.includes('phone');
  }, [deviceType]);

  // 计算响应式尺寸
  const responsive = {
    width: dimensions.width,
    height: dimensions.height,
    scale: dimensions.scale,
    deviceType,
    orientation,
    isTablet,
    isPhone,

    // 响应式值计算
    wp: (percentage: number) => (dimensions.width * percentage) / 100,
    hp: (percentage: number) => (dimensions.height * percentage) / 100,
    min: (...values: number[]) => Math.min(...values),
    max: (...values: number[]) => Math.max(...values),
  };

  return responsive;
};

// 使用示例
const ResponsiveComponent = () => {
  const { wp, hp, deviceType, isTablet, orientation } = useResponsive();

  return (
    <View style={{
      padding: isTablet ? 24 : 16,
      width: wp(100),
      height: orientation === 'portrait' ? hp(60) : hp(40),
    }}>
      <Text style={{
        fontSize: isTablet ? 18 : 16,
      }}>
        响应式内容
      </Text>
    </View>
  );
};
```

---

## 📐 布局系统设计

### 1. 网格系统

```typescript
// 网格系统配置
const GridConfig = {
  // 网格列数
  COLUMNS: {
    PHONE: 12,
    TABLET: 16,
    DESKTOP: 24,
  },

  // 网格间距
  GUTTER: {
    PHONE: 8,
    TABLET: 12,
    DESKTOP: 16,
  },

  // 容器最大宽度
  MAX_WIDTH: {
    PHONE: '100%',
    TABLET: 768,
    DESKTOP: 1200,
  },
};

// 网格组件
interface GridProps {
  children: React.ReactNode;
  columns?: number;
  spacing?: number;
  maxContentWidth?: number;
}

const Grid: React.FC<GridProps> = ({
  children,
  columns,
  spacing,
  maxContentWidth,
}) => {
  const { deviceType, width } = useResponsive();

  const gridColumns = columns || GridConfig.COLUMNS[deviceType.toUpperCase() as keyof typeof GridConfig.COLUMNS];
  const gridSpacing = spacing || GridConfig.GUTTER[deviceType.toUpperCase() as keyof typeof GridConfig.GUTTER];

  const containerStyle = useMemo(() => ({
    flexDirection: 'row' as const,
    flexWrap: 'wrap' as const,
    justifyContent: 'space-between',
    paddingHorizontal: gridSpacing,
    maxWidth: maxContentWidth || '100%',
    alignSelf: 'center',
    width: '100%',
  }), [gridSpacing, maxContentWidth]);

  return (
    <View style={containerStyle}>
      {React.Children.map(children, (child, index) => (
        <GridItem
          key={index}
          columns={gridColumns}
          spacing={gridSpacing}
        >
          {child}
        </GridItem>
      ))}
    </View>
  );
};

interface GridItemProps {
  children: React.ReactNode;
  span?: number;
  columns: number;
  spacing: number;
}

const GridItem: React.FC<GridItemProps> = ({
  children,
  span = 1,
  columns,
  spacing,
}) => {
  const itemWidth = useMemo(() => {
    const totalSpacing = spacing * (columns - 1);
    const availableWidth = `100% - ${totalSpacing}px`;
    const itemWidth = `(${availableWidth} / ${columns}) * ${span}`;
    return `calc(${itemWidth})`;
  }, [span, columns, spacing]);

  const itemStyle = useMemo(() => ({
    width: itemWidth,
    marginBottom: spacing,
  }), [itemWidth, spacing]);

  return (
    <View style={itemStyle}>
      {children}
    </View>
  );
};
```

### 2. Flexbox布局适配

```typescript
// 响应式Flexbox工具
const useFlexResponsive = () => {
  const { deviceType, isTablet, orientation } = useResponsive();

  // 响应式Flex属性
  const getFlexDirection = useCallback((portrait: FlexDirection, landscape: FlexDirection) => {
    return orientation === 'portrait' ? portrait : landscape;
  }, [orientation]);

  const getFlexWrap = useCallback((phone: FlexWrap, tablet: FlexWrap) => {
    return isTablet ? tablet : phone;
  }, [isTablet]);

  const getJustifyContent = useCallback((phone: JustifyContent, tablet: JustifyContent) => {
    return isTablet ? tablet : phone;
  }, [isTablet]);

  const getAlignItems = useCallback((phone: AlignItems, tablet: AlignItems) => {
    return isTablet ? tablet : phone;
  }, [isTablet]);

  return {
    getFlexDirection,
    getFlexWrap,
    getJustifyContent,
    getAlignItems,
  };
};

// 响应式布局Hook
const useLayout = () => {
  const { wp, hp, deviceType, isTablet, orientation } = useResponsive();

  // 容器布局
  const containerStyle = useMemo(() => ({
    flex: 1,
    paddingHorizontal: isTablet ? 24 : 16,
    paddingVertical: isTablet ? 20 : 16,
  }), [isTablet]);

  // 卡片布局
  const cardStyle = useMemo(() => ({
    padding: isTablet ? 20 : 16,
    borderRadius: isTablet ? 16 : 12,
    marginBottom: isTablet ? 16 : 12,
    shadowOffset: { width: 0, height: isTablet ? 4 : 2 },
    shadowOpacity: isTablet ? 0.15 : 0.1,
    shadowRadius: isTablet ? 8 : 4,
    elevation: isTablet ? 6 : 3,
  }), [isTablet]);

  // 列表项布局
  const listItemStyle = useMemo(() => ({
    paddingVertical: isTablet ? 16 : 12,
    paddingHorizontal: isTablet ? 20 : 16,
    minHeight: isTablet ? 80 : 60,
  }), [isTablet]);

  return {
    containerStyle,
    cardStyle,
    listItemStyle,

    // 响应式尺寸
    headerHeight: orientation === 'portrait' ? hp(8) : hp(12),
    tabBarHeight: isTablet ? 80 : 60,
    contentSpacing: isTablet ? 20 : 16,
  };
};
```

### 3. 组件自适应

```typescript
// 自适应卡片组件
const ResponsiveCard = ({
  title,
  subtitle,
  children,
  style,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  style?: ViewStyle;
}) => {
  const { deviceType, isTablet } = useResponsive();
  const { cardStyle } = useLayout();

  const cardLayout = useMemo(() => ({
    ...cardStyle,
    backgroundColor: '#FFFFFF',
    width: isTablet ? '48%' : '100%',
  }), [cardStyle, isTablet]);

  const titleStyle = useMemo(() => ({
    fontSize: isTablet ? 20 : 18,
    fontWeight: 'bold' as const,
    marginBottom: isTablet ? 8 : 4,
  }), [isTablet]);

  const subtitleStyle = useMemo(() => ({
    fontSize: isTablet ? 16 : 14,
    color: '#6B7280',
    marginBottom: isTablet ? 16 : 12,
  }), [isTablet]);

  return (
    <View style={[cardLayout, style]}>
      <Text style={titleStyle}>{title}</Text>
      {subtitle && <Text style={subtitleStyle}>{subtitle}</Text>}
      {children}
    </View>
  );
};

// 自适应列表组件
const ResponsiveList = ({
  data,
  renderItem,
  keyExtractor,
  numColumns,
}: {
  data: any[];
  renderItem: (item: any, index: number) => React.ReactElement;
  keyExtractor: (item: any, index: number) => string;
  numColumns?: number;
}) => {
  const { deviceType, isTablet } = useResponsive();

  const columns = numColumns || (isTablet ? 2 : 1);

  return (
    <FlatList
      data={data}
      renderItem={({ item, index }) => (
        <View style={{ flex: 1 / columns, margin: isTablet ? 8 : 4 }}>
          {renderItem(item, index)}
        </View>
      )}
      keyExtractor={keyExtractor}
      numColumns={columns}
      contentContainerStyle={{
        paddingHorizontal: isTablet ? 16 : 12,
        paddingBottom: isTablet ? 24 : 16,
      }}
      showsVerticalScrollIndicator={false}
    />
  );
};
```

---

## 📝 字体系统设计

### 1. 响应式字体配置

```typescript
// 字体系统配置
const TypographyConfig = {
  // 基础字体大小（基于@2x屏幕）
  BASE_FONT_SIZE: 16,

  // 响应式缩放比例
  SCALE_FACTORS: {
    PHONE_SMALL: 0.9,
    PHONE_MEDIUM: 1.0,
    TABLET_SMALL: 1.1,
    TABLET_MEDIUM: 1.2,
    TABLET_LARGE: 1.3,
  },

  // 字体大小等级
  FONT_SIZES: {
    XS: 12,
    SM: 14,
    MD: 16,
    LG: 18,
    XL: 20,
    XXL: 24,
    XXXL: 32,
  },

  // 行高配置
  LINE_HEIGHTS: {
    TIGHT: 1.2,
    NORMAL: 1.4,
    RELAXED: 1.6,
    LOOSE: 1.8,
  },

  // 字重配置
  FONT_WEIGHTS: {
    LIGHT: '300' as const,
    NORMAL: '400' as const,
    MEDIUM: '500' as const,
    SEMIBOLD: '600' as const,
    BOLD: '700' as const,
  },
};

// 响应式字体Hook
const useTypography = () => {
  const { deviceType } = useResponsive();

  const scaleFactor = TypographyConfig.SCALE_FACTORS[
    deviceType.toUpperCase() as keyof typeof TypographyConfig.SCALE_FACTORS
  ];

  // 计算响应式字体大小
  const getFontSize = useCallback((baseSize: number) => {
    return Math.round(baseSize * scaleFactor);
  }, [scaleFactor]);

  // 字体样式生成器
  const createTextStyle = useCallback((
    size: keyof typeof TypographyConfig.FONT_SIZES,
    weight: keyof typeof TypographyConfig.FONT_WEIGHTS = 'NORMAL',
    lineHeight: keyof typeof TypographyConfig.LINE_HEIGHTS = 'NORMAL'
  ) => {
    const baseSize = TypographyConfig.FONT_SIZES[size];
    const responsiveSize = getFontSize(baseSize);
    const lineHeightValue = TypographyConfig.LINE_HEIGHTS[lineHeight];

    return {
      fontSize: responsiveSize,
      fontWeight: TypographyConfig.FONT_WEIGHTS[weight],
      lineHeight: responsiveSize * lineHeightValue,
    };
  }, [getFontSize]);

  return {
    createTextStyle,
    scaleFactor,
    fontSizes: {
      xs: createTextStyle('XS'),
      sm: createTextStyle('SM'),
      md: createTextStyle('MD'),
      lg: createTextStyle('LG'),
      xl: createTextStyle('XL'),
      xxl: createTextStyle('XXL'),
      xxxl: createTextStyle('XXXL'),
    },
  };
};

// 预定义文本组件
const ResponsiveText = ({
  children,
  variant = 'md',
  weight = 'normal',
  color = '#111827',
  style,
  numberOfLines,
}: {
  children: React.ReactNode;
  variant?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl' | 'xxxl';
  weight?: 'light' | 'normal' | 'medium' | 'semibold' | 'bold';
  color?: string;
  style?: TextStyle;
  numberOfLines?: number;
}) => {
  const { createTextStyle } = useTypography();

  const textStyle = useMemo(() => ({
    ...createTextStyle(
      variant.toUpperCase() as keyof typeof TypographyConfig.FONT_SIZES,
      weight.toUpperCase() as keyof typeof TypographyConfig.FONT_WEIGHTS
    ),
    color,
  }), [createTextStyle, variant, weight, color]);

  return (
    <Text
      style={[textStyle, style]}
      numberOfLines={numberOfLines}
    >
      {children}
    </Text>
  );
};
```

### 2. 动态字体大小

```typescript
// 动态字体大小管理
const useDynamicFontSize = () => {
  const [fontScale, setFontScale] = useState(1);

  useEffect(() => {
    // 监听系统字体大小变化
    const subscription = PixelRatio.addFontScaleListener((scale) => {
      setFontScale(scale);
    });

    // 获取当前字体缩放比例
    setFontScale(PixelRatio.getFontScale());

    return () => {
      subscription?.remove();
    };
  }, []);

  const getScaledFontSize = useCallback((baseSize: number) => {
    return Math.round(baseSize * fontScale);
  }, [fontScale]);

  return {
    fontScale,
    getScaledFontSize,
  };
};

// 可访问性字体组件
const AccessibleText = ({
  children,
  minScale = 0.8,
  maxScale = 2.0,
  ...props
}: {
  children: React.ReactNode;
  minScale?: number;
  maxScale?: number;
} & TextProps) => {
  const { getScaledFontSize, fontScale } = useDynamicFontSize();

  const adjustedStyle = useMemo(() => {
    if (!props.style) return props.style;

    const fontSize = (props.style as TextStyle).fontSize;
    if (typeof fontSize !== 'number') return props.style;

    const scaledSize = getScaledFontSize(fontSize);
    const clampedSize = Math.max(
      fontSize * minScale,
      Math.min(scaledSize, fontSize * maxScale)
    );

    return {
      ...props.style,
      fontSize: clampedSize,
    };
  }, [props.style, getScaledFontSize, minScale, maxScale, fontScale]);

  return (
    <Text {...props} style={adjustedStyle}>
      {children}
    </Text>
  );
};
```

---

## 🖼️ 图片与图标适配

### 1. 响应式图片处理

```typescript
// 图片尺寸配置
const ImageConfig = {
  // 不同设备的图片尺寸
  THUMBNAIL_SIZES: {
    PHONE_SMALL: { width: 80, height: 80 },
    PHONE_MEDIUM: { width: 100, height: 100 },
    TABLET_SMALL: { width: 120, height: 120 },
    TABLET_MEDIUM: { width: 140, height: 140 },
  },

  // 预览图片尺寸
  PREVIEW_SIZES: {
    PHONE_SMALL: { width: 300, height: 200 },
    PHONE_MEDIUM: { width: 350, height: 233 },
    TABLET_SMALL: { width: 400, height: 267 },
    TABLET_MEDIUM: { width: 500, height: 333 },
  },

  // 全屏图片尺寸
  FULLSCREEN_SIZES: {
    PHONE_SMALL: { width: 375, height: 667 },
    PHONE_MEDIUM: { width: 414, height: 896 },
    TABLET_SMALL: { width: 768, height: 1024 },
    TABLET_MEDIUM: { width: 1024, height: 1366 },
  },
};

// 响应式图片组件
const ResponsiveImage = ({
  source,
  type = 'preview',
  style,
  resizeMode = 'cover',
  onLoad,
  onError,
}: {
  source: ImageSourcePropType;
  type?: 'thumbnail' | 'preview' | 'fullscreen';
  style?: ImageStyle;
  resizeMode?: ResizeMode;
  onLoad?: (info: ImageLoadSuccessInfo) => void;
  onError?: (error: ImageLoadError) => void;
}) => {
  const { deviceType } = useResponsive();

  const imageSize = useMemo(() => {
    const config = ImageConfig[`${type.toUpperCase()}_SIZES` as keyof typeof ImageConfig];
    return config[deviceType.toUpperCase() as keyof typeof config];
  }, [deviceType, type]);

  const imageStyle = useMemo(() => ({
    width: imageSize.width,
    height: imageSize.height,
    ...style,
  }), [imageSize, style]);

  return (
    <Image
      source={source}
      style={imageStyle}
      resizeMode={resizeMode}
      onLoad={onLoad}
      onError={onError}
    />
  );
};

// 自适应背景图片
const AdaptiveBackground = ({
  children,
  source,
  overlay = true,
  style,
}: {
  children: React.ReactNode;
  source: ImageSourcePropType;
  overlay?: boolean;
  style?: ViewStyle;
}) => {
  const { width, height } = useResponsive();

  return (
    <View style={[{ flex: 1 }, style]}>
      <Image
        source={source}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: width,
          height: height,
          resizeMode: 'cover',
        }}
      />
      {overlay && (
        <View
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: width,
            height: height,
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
          }}
        />
      )}
      <View style={{ flex: 1 }}>{children}</View>
    </View>
  );
};
```

### 2. 响应式图标系统

```typescript
// 图标尺寸配置
const IconConfig = {
  // 图标尺寸
  SIZES: {
    XXS: 12,
    XS: 16,
    SM: 20,
    MD: 24,
    LG: 32,
    XL: 40,
    XXL: 48,
    XXXL: 64,
  },

  // 响应式图标映射
  RESPONSIVE_SIZES: {
    PHONE: {
      navigation: IconConfig.SIZES.MD,
      action: IconConfig.SIZES.LG,
      display: IconConfig.SIZES.XL,
    },
    TABLET: {
      navigation: IconConfig.SIZES.LG,
      action: IconConfig.SIZES.XL,
      display: IconConfig.SIZES.XXL,
    },
  },
};

// 响应式图标组件
const ResponsiveIcon = ({
  name,
  type = 'action',
  color = '#111827',
  style,
}: {
  name: string;
  type?: 'navigation' | 'action' | 'display';
  color?: string;
  style?: ViewStyle;
}) => {
  const { isTablet } = useResponsive();

  const deviceType = isTablet ? 'TABLET' : 'PHONE';
  const iconSize = IconConfig.RESPONSIVE_SIZES[deviceType][type];

  const iconStyle = useMemo(() => ({
    width: iconSize,
    height: iconSize,
    ...style,
  }), [iconSize, style]);

  return (
    <VectorIcon
      name={name}
      size={iconSize}
      color={color}
      style={iconStyle}
    />
  );
};

// 自适应按钮图标
const AdaptiveIconButton = ({
  icon,
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
}: {
  icon: string;
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'small' | 'medium' | 'large';
}) => {
  const { isTablet } = useResponsive();
  const { createTextStyle } = useTypography();

  const buttonStyle = useMemo(() => {
    const baseStyle = {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
      justifyContent: 'center' as const,
      borderRadius: isTablet ? 12 : 8,
      paddingHorizontal: size === 'small' ? 12 : size === 'large' ? 24 : 16,
      paddingVertical: size === 'small' ? 8 : size === 'large' ? 16 : 12,
      minWidth: isTablet ? 120 : 100,
    };

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          backgroundColor: '#3B82F6',
        };
      case 'secondary':
        return {
          ...baseStyle,
          backgroundColor: '#F3F4F6',
          borderWidth: 1,
          borderColor: '#D1D5DB',
        };
      case 'ghost':
        return {
          ...baseStyle,
          backgroundColor: 'transparent',
        };
      default:
        return baseStyle;
    }
  }, [isTablet, size, variant]);

  const iconSize = size === 'small' ? IconConfig.SIZES.SM : size === 'large' ? IconConfig.SIZES.LG : IconConfig.SIZES.MD;

  const textStyle = createTextStyle(
    size === 'small' ? 'SM' : size === 'large' ? 'LG' : 'MD',
    'MEDIUM'
  );

  const textColor = variant === 'primary' ? '#FFFFFF' : '#111827';

  return (
    <TouchableOpacity style={buttonStyle} onPress={onPress}>
      <VectorIcon
        name={icon}
        size={iconSize}
        color={textColor}
        style={{ marginRight: 8 }}
      />
      <Text style={[textStyle, { color: textColor }]}>
        {title}
      </Text>
    </TouchableOpacity>
  );
};
```

---

## 🎨 主题与样式适配

### 1. 响应式主题系统

```typescript
// 主题配置
const ThemeConfig = {
  // 间距系统
  SPACING: {
    XS: 4,
    SM: 8,
    MD: 16,
    LG: 24,
    XL: 32,
    XXL: 48,
    XXXL: 64,
  },

  // 响应式间距倍数
  SPACING_MULTIPLIERS: {
    PHONE: 1.0,
    TABLET: 1.5,
  },

  // 圆角系统
  BORDER_RADIUS: {
    SM: 4,
    MD: 8,
    LG: 12,
    XL: 16,
    XXL: 24,
    ROUND: 9999,
  },

  // 响应式圆角
  RESPONSIVE_BORDER_RADIUS: {
    PHONE: {
      button: 8,
      card: 12,
      modal: 16,
    },
    TABLET: {
      button: 12,
      card: 16,
      modal: 20,
    },
  },
};

// 响应式主题Hook
const useResponsiveTheme = () => {
  const { isTablet } = useResponsive();

  const spacingMultiplier = ThemeConfig.SPACING_MULTIPLIERS[isTablet ? 'TABLET' : 'PHONE'];

  // 计算响应式间距
  const spacing = useMemo(() => {
    const createSpacing = (baseSize: number) => Math.round(baseSize * spacingMultiplier);

    return {
      xs: createSpacing(ThemeConfig.SPACING.XS),
      sm: createSpacing(ThemeConfig.SPACING.SM),
      md: createSpacing(ThemeConfig.SPACING.MD),
      lg: createSpacing(ThemeConfig.SPACING.LG),
      xl: createSpacing(ThemeConfig.SPACING.XL),
      xxl: createSpacing(ThemeConfig.SPACING.XXL),
      xxxl: createSpacing(ThemeConfig.SPACING.XXXL),
    };
  }, [spacingMultiplier]);

  // 响应式圆角
  const borderRadius = useMemo(() => {
    const deviceType = isTablet ? 'TABLET' : 'PHONE';
    return ThemeConfig.RESPONSIVE_BORDER_RADIUS[deviceType];
  }, [isTablet]);

  return {
    spacing,
    borderRadius,
  };
};

// 响应式样式生成器
const useStyles = (styleFactory: (theme: any) => any) => {
  const theme = useResponsiveTheme();

  return useMemo(() => styleFactory(theme), [theme, styleFactory]);
};

// 使用示例
const ResponsiveComponent = () => {
  const styles = useStyles((theme) => ({
    container: {
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.card,
      backgroundColor: '#FFFFFF',
    },
    button: {
      paddingVertical: theme.spacing.sm,
      paddingHorizontal: theme.spacing.lg,
      borderRadius: theme.borderRadius.button,
    },
  }));

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.button}>
        <Text>响应式按钮</Text>
      </TouchableOpacity>
    </View>
  );
};
```

### 2. 暗色模式适配

```typescript
// 暗色模式主题
const DarkTheme = {
  colors: {
    primary: '#60A5FA',
    background: '#111827',
    surface: '#1F2937',
    text: '#F9FAFB',
    textSecondary: '#D1D5DB',
    border: '#374151',
    error: '#F87171',
    warning: '#FBBF24',
    success: '#34D399',
  },
};

// 亮色模式主题
const LightTheme = {
  colors: {
    primary: '#3B82F6',
    background: '#FFFFFF',
    surface: '#F9FAFB',
    text: '#111827',
    textSecondary: '#6B7280',
    border: '#E5E7EB',
    error: '#EF4444',
    warning: '#F59E0B',
    success: '#10B981',
  },
};

// 主题Hook
const useTheme = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // 检查系统主题偏好
    const subscription = Appearance.addChangeListener(({ colorScheme }) => {
      setIsDarkMode(colorScheme === 'dark');
    });

    // 获取当前主题
    setIsDarkMode(Appearance.getColorScheme() === 'dark');

    return () => subscription?.remove();
  }, []);

  const theme = isDarkMode ? DarkTheme : LightTheme;

  const toggleTheme = useCallback(() => {
    setIsDarkMode(!isDarkMode);
  }, [isDarkMode]);

  return {
    theme,
    isDarkMode,
    toggleTheme,
  };
};

// 响应式主题组件
const ThemedView = ({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: ViewStyle;
}) => {
  const { theme } = useTheme();
  const { spacing, borderRadius } = useResponsiveTheme();

  const themedStyle = useMemo(() => ({
    backgroundColor: theme.colors.background,
    borderColor: theme.colors.border,
    ...style,
  }), [theme, style]);

  return (
    <View style={themedStyle}>
      {children}
    </View>
  );
};
```

---

## ⚡ 性能优化策略

### 1. 渲染优化

```typescript
// 性能优化的响应式组件
const OptimizedResponsiveComponent = React.memo<{
  data: any[];
  renderItem: (item: any, index: number) => React.ReactElement;
}>(({ data, renderItem }) => {
  const { deviceType, isTablet } = useResponsive();

  // 懒加载配置
  const getItemLayout = useCallback((data: any, index: number) => ({
    length: isTablet ? 120 : 80,
    offset: (isTablet ? 120 : 80) * index,
    index,
  }), [isTablet]);

  // 关键项提取
  const keyExtractor = useCallback((item: any, index: number) => {
    return item.id || `item-${index}`;
  }, []);

  // 渲染项优化
  const optimizedRenderItem = useCallback(({ item, index }: { item: any; index: number }) => {
    return (
      <MemoizedListItem item={item} index={index}>
        {renderItem(item, index)}
      </MemoizedListItem>
    );
  }, [renderItem]);

  return (
    <FlatList
      data={data}
      renderItem={optimizedRenderItem}
      keyExtractor={keyExtractor}
      getItemLayout={getItemLayout}
      removeClippedSubviews={true}
      maxToRenderPerBatch={isTablet ? 10 : 5}
      updateCellsBatchingPeriod={isTabled ? 50 : 100}
      initialNumToRender={isTablet ? 15 : 10}
      windowSize={isTablet ? 15 : 10}
    />
  );
});

// 记忆化列表项
const MemoizedListItem = React.memo<{
  item: any;
  index: number;
  children: React.ReactNode;
}>(({ children }) => {
  return <>{children}</>;
});
```

### 2. 图片优化

```typescript
// 图片优化管理器
class ImageOptimizationManager {
  private static instance: ImageOptimizationManager;
  private imageCache: Map<string, any> = new Map();
  private loadingPromises: Map<string, Promise<any>> = new Map();

  static getInstance(): ImageOptimizationManager {
    if (!ImageOptimizationManager.instance) {
      ImageOptimizationManager.instance = new ImageOptimizationManager();
    }
    return ImageOptimizationManager.instance;
  }

  // 获取优化的图片URL
  getOptimizedImageUrl(
    baseUrl: string,
    targetWidth: number,
    targetHeight: number,
    quality: number = 80
  ): string {
    const cacheKey = `${baseUrl}_${targetWidth}_${targetHeight}_${quality}`;

    if (this.imageCache.has(cacheKey)) {
      return this.imageCache.get(cacheKey);
    }

    // 根据设备像素密度调整尺寸
    const scale = PixelRatio.get();
    const scaledWidth = Math.round(targetWidth * scale);
    const scaledHeight = Math.round(targetHeight * scale);

    // 构建优化后的URL（假设后端支持图片处理参数）
    const optimizedUrl = `${baseUrl}?w=${scaledWidth}&h=${scaledHeight}&q=${quality}&format=webp`;

    this.imageCache.set(cacheKey, optimizedUrl);
    return optimizedUrl;
  }

  // 预加载图片
  async preloadImage(url: string): Promise<void> {
    if (this.loadingPromises.has(url)) {
      return this.loadingPromises.get(url);
    }

    const promise = new Promise<void>((resolve, reject) => {
      Image.prefetch(url)
        .then(() => resolve())
        .catch((error) => reject(error));
    });

    this.loadingPromises.set(url, promise);
    return promise;
  }

  // 清理缓存
  clearCache(): void {
    this.imageCache.clear();
    this.loadingPromises.clear();
  }
}

// 优化的响应式图片组件
const OptimizedResponsiveImage = ({
  source,
  type = 'preview',
  style,
  placeholder,
  ...props
}: {
  source: { uri: string };
  type?: 'thumbnail' | 'preview' | 'fullscreen';
  style?: ImageStyle;
  placeholder?: React.ReactNode;
}) => {
  const { deviceType } = useResponsive();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  const imageManager = ImageOptimizationManager.getInstance();

  const imageSize = useMemo(() => {
    const config = ImageConfig[`${type.toUpperCase()}_SIZES` as keyof typeof ImageConfig];
    return config[deviceType.toUpperCase() as keyof typeof config];
  }, [deviceType, type]);

  const optimizedSource = useMemo(() => {
    if (!source.uri) return source;

    const optimizedUrl = imageManager.getOptimizedImageUrl(
      source.uri,
      imageSize.width,
      imageSize.height,
      type === 'thumbnail' ? 60 : 80
    );

    return { uri: optimizedUrl };
  }, [source.uri, imageSize.width, imageSize.height, type, imageManager]);

  // 预加载图片
  useEffect(() => {
    if (optimizedSource.uri) {
      imageManager.preloadImage(optimizedSource.uri);
    }
  }, [optimizedSource.uri, imageManager]);

  const handleLoad = useCallback(() => {
    setIsLoading(false);
    setError(false);
  }, []);

  const handleError = useCallback(() => {
    setIsLoading(false);
    setError(true);
  }, []);

  const imageStyle = useMemo(() => ({
    width: imageSize.width,
    height: imageSize.height,
    ...style,
  }), [imageSize, style]);

  return (
    <View style={imageStyle}>
      {isLoading && placeholder && placeholder}
      <Image
        source={optimizedSource}
        style={[
          StyleSheet.absoluteFillObject,
          { opacity: isLoading ? 0 : 1 }
        ]}
        onLoad={handleLoad}
        onError={handleError}
        {...props}
      />
      {error && (
        <View style={StyleSheet.absoluteFillObject}>
          <ImageErrorFallback width={imageSize.width} height={imageSize.height} />
        </View>
      )}
    </View>
  );
};
```

---

## 🧪 测试策略

### 1. 响应式测试工具

```typescript
// 响应式测试工具
const ResponsiveTestUtils = {
  // 模拟不同设备尺寸
  mockDimensions: (width: number, height: number, scale: number = 2) => {
    Dimensions.set({
      window: { width, height, scale, fontScale: scale },
      screen: { width, height, scale, fontScale: scale },
    });
  },

  // 模拟设备类型
  mockDeviceType: (type: 'phone' | 'tablet') => {
    const mockDimensions = type === 'phone'
      ? { width: 375, height: 812, scale: 3 }
      : { width: 768, height: 1024, scale: 2 };

    ResponsiveTestUtils.mockDimensions(
      mockDimensions.width,
      mockDimensions.height,
      mockDimensions.scale
    );
  },

  // 验证响应式样式
  expectResponsiveStyle: (component: React.ReactElement, expectedBreakpoint: string) => {
    const { getByTestId } = render(component);
    const element = getByTestId('responsive-element');

    // 验证样式是否符合预期断点
    expect(element).toHaveStyle({
      fontSize: expectedBreakpoint === 'phone' ? 16 : 18,
    });
  },
};

// 响应式测试用例
describe('Responsive Design', () => {
  beforeEach(() => {
    // 重置模拟尺寸
    ResponsiveTestUtils.mockDimensions(375, 812, 3);
  });

  test('should adapt layout for phone devices', () => {
    ResponsiveTestUtils.mockDeviceType('phone');

    const { result } = renderHook(() => useResponsive());

    expect(result.current.deviceType).toBe(DeviceType.PHONE_MEDIUM);
    expect(result.current.isPhone).toBe(true);
    expect(result.current.isTablet).toBe(false);
  });

  test('should adapt layout for tablet devices', () => {
    ResponsiveTestUtils.mockDeviceType('tablet');

    const { result } = renderHook(() => useResponsive());

    expect(result.current.deviceType).toBe(DeviceType.TABLET_SMALL);
    expect(result.current.isPhone).toBe(false);
    expect(result.current.isTablet).toBe(true);
  });

  test('should handle orientation changes', () => {
    // 竖屏
    ResponsiveTestUtils.mockDimensions(375, 812, 3);
    const { result: portraitResult } = renderHook(() => useResponsive());
    expect(portraitResult.current.orientation).toBe(Orientation.PORTRAIT);

    // 横屏
    ResponsiveTestUtils.mockDimensions(812, 375, 3);
    const { result: landscapeResult } = renderHook(() => useResponsive());
    expect(landscapeResult.current.orientation).toBe(Orientation.LANDSCAPE);
  });

  test('should scale fonts appropriately', () => {
    ResponsiveTestUtils.mockDeviceType('tablet');

    const { result } = renderHook(() => useTypography());

    const textStyle = result.current.createTextStyle('MD', 'NORMAL');
    expect(textStyle.fontSize).toBeGreaterThan(16); // 基础字体大小
  });

  test('should optimize images for different devices', () => {
    const imageManager = ImageOptimizationManager.getInstance();

    // 手机设备
    ResponsiveTestUtils.mockDeviceType('phone');
    const phoneUrl = imageManager.getOptimizedImageUrl(
      'https://example.com/image.jpg',
      350,
      233
    );

    // 平板设备
    ResponsiveTestUtils.mockDeviceType('tablet');
    const tabletUrl = imageManager.getOptimizedImageUrl(
      'https://example.com/image.jpg',
      500,
      333
    );

    expect(phoneUrl).not.toBe(tabletUrl);
    expect(phoneUrl).toContain('w=700'); // 350 * 2
    expect(tabletUrl).toContain('w=1000'); // 500 * 2
  });
});
```

### 2. 性能测试

```typescript
// 响应式性能测试
describe('Responsive Performance', () => {
  test('should not cause unnecessary re-renders', async () => {
    const renderCount = jest.fn();

    const TestComponent = React.memo(() => {
      const { deviceType } = useResponsive();
      renderCount();
      return <Text>Device: {deviceType}</Text>;
    });

    const { rerender } = render(<TestComponent />);

    // 初始渲染
    expect(renderCount).toHaveBeenCalledTimes(1);

    // 重新渲染（没有props变化）
    rerender(<TestComponent />);

    // 不应该有额外渲染
    expect(renderCount).toHaveBeenCalledTimes(1);
  });

  test('should efficiently handle dimension changes', async () => {
    const startTime = performance.now();

    const TestComponent = () => {
      const responsive = useResponsive();
      return <Text>Width: {responsive.width}</Text>;
    };

    render(<TestComponent />);

    // 模拟快速尺寸变化
    for (let i = 0; i < 10; i++) {
      ResponsiveTestUtils.mockDimensions(375 + i, 812, 3);
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    const endTime = performance.now();
    const duration = endTime - startTime;

    // 尺寸变化应该在合理时间内完成
    expect(duration).toBeLessThan(1000);
  });
});
```

---

## 📚 相关文档

### 架构文档系列
- [01-架构概述](01-overview.md)：详细的架构设计说明
- [02-技术架构](02-technical-architecture.md)：技术栈和架构模式
- [03-组件设计](03-component-design.md)：组件设计规范
- [04-API集成](04-api-integration.md)：API接口集成方案
- [05-状态管理](05-state-management.md)：状态管理架构
- [06-导航设计](06-navigation-design.md)：导航系统设计

### 设计文档系列
- [08-性能优化](08-performance-optimization.md)：性能优化策略

### 开发文档系列
- [09-测试策略](09-testing-strategy.md)：测试策略和工具
- [10-部署指南](10-deployment-guide.md)：应用打包和发布

### 技术参考
- [React Native响应式设计](https://reactnative.dev/docs/layout-animations)
- [React Native Flexbox布局](https://reactnative.dev/docs/flexbox)
- [移动端设计规范](https://developer.apple.com/design/human-interface-guidelines/)
- [Material Design响应式设计](https://material.io/design/layout/responsive-layout-grid.html)

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**下次更新**: 根据不同设备测试结果和用户反馈持续优化