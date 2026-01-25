# Dehaze React Native - 组件设计规范

**文档版本**: v1.0
**最后更新**: 2025-11-22
**项目名称**: dehaze-react-native
**目标平台**: iOS、Android

---

## 📋 文档概述

本文档详细描述了Dehaze React
Native应用的组件设计规范，包括组件架构原则、通用组件设计、业务组件实现和最佳实践。基于[demo中的设计规范](../../../docs/08-UI-UX设计规范.md)
和移动端特性，提供可复用、高性能的组件设计方案。

---

## 🎨 组件设计原则

### 1. 设计原则

#### 单一职责原则 (Single Responsibility Principle)
每个组件只负责一个功能领域，保持组件的专注性和可维护性。

```typescript
// ✅ 好的设计 - 职责明确
const ImageUploadButton = ({ onUpload, maxSize }: ImageUploadProps) => {
  // 只负责图片上传按钮的UI和交互
};

const ImagePreview = ({ imageUri, onClose }: ImagePreviewProps) => {
  // 只负责图片预览功能
};

// ❌ 避免的设计 - 职责混乱
const ImageUploadAndPreview = ({ onUpload, maxSize, imageUri }: CombinedProps) => {
  // 同时负责上传和预览，违反单一职责原则
};
```

#### 可复用性原则 (Reusability)
组件设计应该高度可复用，通过props控制行为和样式。

```typescript
// ✅ 高度可复用的按钮组件
interface ButtonProps {
  title: string;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
}

const Button: React.FC<ButtonProps> = ({
  title,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  onPress,
  style,
}) => {
  // 通过props控制不同的外观和行为
};
```

#### 组合优于继承原则 (Composition over Inheritance)
使用组件组合而不是继承来实现功能扩展。

```typescript
// ✅ 使用组合
const Card = ({ children, style, ...props }: CardProps) => (
  <View style={[styles.card, style]} {...props}>
    {children}
  </View>
);

const ImageCard = ({ image, ...props }: ImageCardProps) => (
  <Card {...props}>
    <Image source={{ uri: image.uri }} style={styles.image} />
    <Text>{image.title}</Text>
  </Card>
);

// ❌ 避免继承
class BaseCard extends React.Component {
  // 避免使用继承来实现功能扩展
}
```

### 2. 组件分类体系

#### 按功能层级分类

```typescript
// 基础组件层 (Foundation Layer)
import { View, Text, Image, TouchableOpacity } from 'react-native';

// 通用组件层 (Common Components Layer)
export { Button, Input, Card, Modal, Loading } from '@components/common';

// 业务组件层 (Business Components Layer)
export { ImagePicker, AlgorithmSelector, ProcessingProgress } from '@components/business';

// 页面组件层 (Screen Components Layer)
export { HomeScreen, ImageInputScreen, ProcessingScreen } from '@screens';
```

#### 按复用程度分类

```typescript
// 全局通用组件 (Global Common)
const Button = () => {}; // 在整个应用中通用
const Input = () => {}; // 表单输入组件
const Modal = () => {}; // 弹窗组件

// 功能域组件 (Feature Domain)
const ImageCard = () => {}; // 图片相关功能通用
const AlgorithmCard = () => {}; // 算法相关功能通用

// 页面特定组件 (Page Specific)
const HomeHero = () => {}; // 首页专用
const ProcessingControls = () => {}; // 处理页面专用
```

---

## 🧩 通用组件设计

### 1. 基础组件库

#### Button 按钮组件

```typescript
// components/common/Button/Button.tsx
import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
  Animated,
} from 'react-native';
import { useTheme } from '@theme';

// 按钮变体类型
export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost';
export type ButtonSize = 'small' | 'medium' | 'large';

interface ButtonProps {
  title: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  onPress?: () => void;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export const Button: React.FC<ButtonProps> = ({
  title,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  onPress,
  style,
  textStyle,
}) => {
  const { theme } = useTheme();
  const scaleValue = React.useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleValue, {
      toValue: 0.95,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleValue, {
      toValue: 1,
      useNativeDriver: true,
    }).start();
  };

  const getButtonStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {
      borderRadius: theme.borderRadius.md,
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: icon ? 'row' : 'column',
    };

    // 尺寸样式
    const sizeStyles = {
      small: {
        paddingHorizontal: theme.spacing.sm,
        paddingVertical: theme.spacing.xs,
        minHeight: 36,
      },
      medium: {
        paddingHorizontal: theme.spacing.md,
        paddingVertical: theme.spacing.sm,
        minHeight: 44,
      },
      large: {
        paddingHorizontal: theme.spacing.lg,
        paddingVertical: theme.spacing.md,
        minHeight: 52,
      },
    };

    // 变体样式
    const variantStyles = {
      primary: {
        backgroundColor: disabled ? theme.colors.disabled : theme.colors.primary,
      },
      secondary: {
        backgroundColor: disabled ? theme.colors.disabled : theme.colors.secondary,
      },
      outline: {
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderColor: disabled ? theme.colors.disabled : theme.colors.primary,
      },
      ghost: {
        backgroundColor: 'transparent',
      },
    };

    return StyleSheet.compose([
      baseStyle,
      sizeStyles[size],
      variantStyles[variant],
      fullWidth && { width: '100%' },
      style,
    ]);
  };

  const getTextStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      fontFamily: theme.typography.fontFamily.medium,
      fontWeight: '600',
    };

    const sizeStyles = {
      small: { fontSize: theme.typography.fontSize.sm },
      medium: { fontSize: theme.typography.fontSize.md },
      large: { fontSize: theme.typography.fontSize.lg },
    };

    const variantStyles = {
      primary: { color: theme.colors.white },
      secondary: { color: theme.colors.white },
      outline: { color: theme.colors.primary },
      ghost: { color: theme.colors.primary },
    };

    return StyleSheet.compose([
      baseStyle,
      sizeStyles[size],
      variantStyles[variant],
      textStyle,
    ]);
  };

  return (
    <TouchableOpacity
      style={getButtonStyle()}
      onPress={onPress}
      disabled={disabled || loading}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={0.8}
    >
      <Animated.View style={{ transform: [{ scale: scaleValue }] }}>
        {loading && (
          <ActivityIndicator
            size="small"
            color={variant === 'outline' || variant === 'ghost'
              ? theme.colors.primary
              : theme.colors.white}
            style={{ marginRight: theme.spacing.xs }}
          />
        )}

        {icon && iconPosition === 'left' && !loading && (
          <View style={{ marginRight: theme.spacing.xs }}>
            {icon}
          </View>
        )}

        <Text style={getTextStyle()}>
          {title}
        </Text>

        {icon && iconPosition === 'right' && (
          <View style={{ marginLeft: theme.spacing.xs }}>
            {icon}
          </View>
        )}
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  // 样式已在动态样式中定义
});
```

#### Input 输入框组件

```typescript
// components/common/Input/Input.tsx
import React, { useState, forwardRef } from 'react';
import {
  TextInput,
  View,
  Text,
  StyleSheet,
  TextInputProps,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { useTheme } from '@theme';

export interface InputProps extends Omit<TextInputProps, 'style'> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  containerStyle?: ViewStyle;
  inputStyle?: TextStyle;
  labelStyle?: TextStyle;
  errorStyle?: TextStyle;
  variant?: 'outlined' | 'filled';
}

export const Input = forwardRef<TextInput, InputProps>(({
  label,
  error,
  helperText,
  leftIcon,
  rightIcon,
  containerStyle,
  inputStyle,
  labelStyle,
  errorStyle,
  variant = 'outlined',
  value,
  onFocus,
  onBlur,
  ...props
}, ref) => {
  const { theme } = useTheme();
  const [isFocused, setIsFocused] = useState(false);

  const handleFocus = (e: any) => {
    setIsFocused(true);
    onFocus?.(e);
  };

  const handleBlur = (e: any) => {
    setIsFocused(false);
    onBlur?.(e);
  };

  const getContainerStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {
      marginVertical: theme.spacing.xs,
    };

    const inputContainerStyle: ViewStyle = {
      flexDirection: 'row',
      alignItems: 'center',
      borderRadius: theme.borderRadius.md,
      borderWidth: 1,
    };

    const variantStyles = {
      outlined: {
        backgroundColor: theme.colors.background,
        borderColor: error
          ? theme.colors.error
          : isFocused
            ? theme.colors.primary
            : theme.colors.border,
      },
      filled: {
        backgroundColor: theme.colors.inputBackground,
        borderColor: 'transparent',
        borderBottomWidth: 2,
        borderBottomColor: error
          ? theme.colors.error
          : isFocused
            ? theme.colors.primary
            : theme.colors.border,
      },
    };

    return StyleSheet.compose([
      baseStyle,
      {
        inputContainer: StyleSheet.compose([
          inputContainerStyle,
          variantStyles[variant],
        ]),
      },
      containerStyle,
    ]).inputContainer;
  };

  const getInputStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      flex: 1,
      fontSize: theme.typography.fontSize.md,
      fontFamily: theme.typography.fontFamily.regular,
      color: theme.colors.text,
      paddingVertical: theme.spacing.md,
      paddingHorizontal: theme.spacing.sm,
    };

    if (leftIcon) {
      baseStyle.paddingLeft = 0;
    }

    if (rightIcon) {
      baseStyle.paddingRight = 0;
    }

    return StyleSheet.compose([baseStyle, inputStyle]);
  };

  const getLabelStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      fontSize: theme.typography.fontSize.sm,
      fontFamily: theme.typography.fontFamily.medium,
      marginBottom: theme.spacing.xs,
      color: error ? theme.colors.error : theme.colors.textSecondary,
    };

    return StyleSheet.compose([baseStyle, labelStyle]);
  };

  const getErrorStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      fontSize: theme.typography.fontSize.xs,
      fontFamily: theme.typography.fontFamily.regular,
      marginTop: theme.spacing.xs,
      color: theme.colors.error,
    };

    return StyleSheet.compose([baseStyle, errorStyle]);
  };

  const getHelperTextStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      fontSize: theme.typography.fontSize.xs,
      fontFamily: theme.typography.fontFamily.regular,
      marginTop: theme.spacing.xs,
      color: theme.colors.textSecondary,
    };

    return baseStyle;
  };

  return (
    <View style={containerStyle}>
      {label && (
        <Text style={getLabelStyle()}>
          {label}
        </Text>
      )}

      <View style={getContainerStyle()}>
        {leftIcon && (
          <View style={{
            paddingHorizontal: theme.spacing.sm,
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {leftIcon}
          </View>
        )}

        <TextInput
          ref={ref}
          style={getInputStyle()}
          value={value}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholderTextColor={theme.colors.textSecondary}
          {...props}
        />

        {rightIcon && (
          <View style={{
            paddingHorizontal: theme.spacing.sm,
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {rightIcon}
          </View>
        )}
      </View>

      {error && (
        <Text style={getErrorStyle()}>
          {error}
        </Text>
      )}

      {!error && helperText && (
        <Text style={getHelperTextStyle()}>
          {helperText}
        </Text>
      )}
    </View>
  );
});

Input.displayName = 'Input';
```

#### Card 卡片组件

```typescript
// components/common/Card/Card.tsx
import React from 'react';
import {
  View,
  StyleSheet,
  ViewStyle,
  TouchableOpacity,
  TouchableOpacityProps,
} from 'react-native';
import { useTheme } from '@theme';

export interface CardProps extends TouchableOpacityProps {
  children: React.ReactNode;
  variant?: 'elevated' | 'outlined' | 'filled';
  padding?: 'none' | 'small' | 'medium' | 'large';
  margin?: 'none' | 'small' | 'medium' | 'large';
  borderRadius?: 'none' | 'small' | 'medium' | 'large';
  onPress?: () => void;
  style?: ViewStyle;
  contentStyle?: ViewStyle;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'elevated',
  padding = 'medium',
  margin = 'none',
  borderRadius = 'medium',
  onPress,
  style,
  contentStyle,
  ...touchableOpacityProps
}) => {
  const { theme } = useTheme();

  const getCardStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {
      backgroundColor: theme.colors.surface,
      overflow: 'hidden',
    };

    // 变体样式
    const variantStyles = {
      elevated: {
        shadowColor: theme.colors.shadow,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
      },
      outlined: {
        borderWidth: 1,
        borderColor: theme.colors.border,
      },
      filled: {
        backgroundColor: theme.colors.surfaceVariant,
      },
    };

    // 内边距样式
    const paddingStyles = {
      none: {},
      small: { padding: theme.spacing.sm },
      medium: { padding: theme.spacing.md },
      large: { padding: theme.spacing.lg },
    };

    // 外边距样式
    const marginStyles = {
      none: {},
      small: { margin: theme.spacing.xs },
      medium: { margin: theme.spacing.sm },
      large: { margin: theme.spacing.md },
    };

    // 圆角样式
    const borderStyles = {
      none: {},
      small: { borderRadius: theme.borderRadius.sm },
      medium: { borderRadius: theme.borderRadius.md },
      large: { borderRadius: theme.borderRadius.lg },
    };

    return StyleSheet.compose([
      baseStyle,
      variantStyles[variant],
      paddingStyles[padding],
      marginStyles[margin],
      borderStyles[borderRadius],
      style,
    ]);
  };

  const CardComponent = onPress ? TouchableOpacity : View;

  return (
    <CardComponent
      style={getCardStyle()}
      onPress={onPress}
      activeOpacity={onPress ? 0.7 : 1}
      {...(onPress ? touchableOpacityProps : {})}
    >
      <View style={contentStyle}>
        {children}
      </View>
    </CardComponent>
  );
};

// CardHeader组件
export const CardHeader: React.FC<{
  title?: string;
  subtitle?: string;
  rightComponent?: React.ReactNode;
  style?: ViewStyle;
}> = ({ title, subtitle, rightComponent, style }) => {
  const { theme } = useTheme();

  return (
    <View style={[
      {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: theme.spacing.sm,
      },
      style,
    ]}>
      <View style={{ flex: 1 }}>
        {title && (
          <Text style={{
            fontSize: theme.typography.fontSize.lg,
            fontFamily: theme.typography.fontFamily.semiBold,
            color: theme.colors.text,
          }}>
            {title}
          </Text>
        )}
        {subtitle && (
          <Text style={{
            fontSize: theme.typography.fontSize.sm,
            fontFamily: theme.typography.fontFamily.regular,
            color: theme.colors.textSecondary,
            marginTop: 2,
          }}>
            {subtitle}
          </Text>
        )}
      </View>
      {rightComponent}
    </View>
  );
};

// CardContent组件
export const CardContent: React.FC<{
  children: React.ReactNode;
  style?: ViewStyle;
}> = ({ children, style }) => {
  return (
    <View style={style}>
      {children}
    </View>
  );
};

// CardActions组件
export const CardActions: React.FC<{
  children: React.ReactNode;
  style?: ViewStyle;
}> = ({ children, style }) => {
  const { theme } = useTheme();

  return (
    <View style={[
      {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        alignItems: 'center',
        marginTop: theme.spacing.md,
        gap: theme.spacing.sm,
      },
      style,
    ]}>
      {children}
    </View>
  );
};
```

### 2. 布局组件

#### Container 容器组件

```typescript
// components/layout/Container/Container.tsx
import React from 'react';
import { View, StyleSheet, ViewStyle, SafeAreaView } from 'react-native';
import { useTheme } from '@theme';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export interface ContainerProps {
  children: React.ReactNode;
  safeArea?: boolean;
  padding?: 'none' | 'small' | 'medium' | 'large';
  maxWidth?: boolean;
  flex?: boolean;
  style?: ViewStyle;
}

export const Container: React.FC<ContainerProps> = ({
  children,
  safeArea = true,
  padding = 'medium',
  maxWidth = true,
  flex = false,
  style,
}) => {
  const { theme } = useTheme();
  const insets = useSafeAreaInsets();

  const getContainerStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {};

    // SafeArea处理
    if (safeArea) {
      baseStyle.paddingTop = insets.top;
      baseStyle.paddingBottom = insets.bottom;
    }

    // 内边距样式
    const paddingStyles = {
      none: {},
      small: { paddingHorizontal: theme.spacing.sm },
      medium: { paddingHorizontal: theme.spacing.md },
      large: { paddingHorizontal: theme.spacing.lg },
    };

    // 最大宽度限制
    if (maxWidth) {
      baseStyle.maxWidth = theme.breakpoints.lg;
      baseStyle.width = '100%';
      baseStyle.alignSelf = 'center';
    }

    // Flex布局
    if (flex) {
      baseStyle.flex = 1;
    }

    return StyleSheet.compose([
      baseStyle,
      paddingStyles[padding],
      style,
    ]);
  };

  const ContainerComponent = safeArea ? SafeAreaView : View;

  return (
    <ContainerComponent style={getContainerStyle()}>
      {children}
    </ContainerComponent>
  );
};
```

#### Grid 网格组件

```typescript
// components/layout/Grid/Grid.tsx
import React from 'react';
import { View, StyleSheet, ViewStyle, Dimensions } from 'react-native';
import { useTheme } from '@theme';

const { width: screenWidth } = Dimensions.get('window');

export interface GridProps {
  children: React.ReactNode;
  columns?: number;
  spacing?: number;
  style?: ViewStyle;
}

export const Grid: React.FC<GridProps> = ({
  children,
  columns = 2,
  spacing = 16,
  style,
}) => {
  const { theme } = useTheme();

  const getGridStyle = (): ViewStyle => ({
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginLeft: -spacing / 2,
    marginRight: -spacing / 2,
    ...style,
  });

  const getItemStyle = (index: number): ViewStyle => {
    const itemWidth = `${100 / columns}%`;

    return {
      width: itemWidth,
      paddingHorizontal: spacing / 2,
      marginBottom: spacing,
    };
  };

  const items = React.Children.toArray(children);

  return (
    <View style={getGridStyle()}>
      {items.map((child, index) => (
        <View key={index} style={getItemStyle(index)}>
          {child}
        </View>
      ))}
    </View>
  );
};

// Grid.Item 组件
export interface GridItemProps {
  children: React.ReactNode;
  span?: number;
  style?: ViewStyle;
}

export const GridItem: React.FC<GridItemProps> = ({
  children,
  span = 1,
  style,
}) => {
  const { theme } = useTheme();

  return (
    <View style={style}>
      {children}
    </View>
  );
};
```

---

## 🎯 业务组件设计

### 1. 图像处理组件

#### ImagePicker 图片选择器

```typescript
// components/business/ImagePicker/ImagePicker.tsx
import React, { useState, useRef } from 'react';
import {
  View,
  TouchableOpacity,
  Image,
  Alert,
  Modal,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { launchImageLibrary, launchCamera } from 'react-native-image-picker';
import { useCamera } from '@hooks/useCamera';
import { usePermissions } from '@hooks/usePermissions';
import { Button } from '@components/common';
import { useTheme } from '@theme';

export interface ImagePickerProps {
  onImageSelected: (image: ImageData) => void;
  maxSize?: number; // 最大文件大小（字节）
  quality?: number; // 图片质量 0-1
  allowCamera?: boolean;
  allowGallery?: boolean;
  style?: ViewStyle;
}

interface ImageData {
  uri: string;
  name: string;
  type: string;
  size: number;
  width: number;
  height: number;
}

export const ImagePicker: React.FC<ImagePickerProps> = ({
  onImageSelected,
  maxSize = 10 * 1024 * 1024, // 10MB
  quality = 0.8,
  allowCamera = true,
  allowGallery = true,
  style,
}) => {
  const { theme } = useTheme();
  const [showModal, setShowModal] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const {
    hasPermission: cameraPermission,
    requestPermission: requestCameraPermission
  } = usePermissions('camera');

  const {
    hasPermission: galleryPermission,
    requestPermission: requestGalleryPermission
  } = usePermissions('gallery');

  // 打开相机
  const openCamera = async () => {
    try {
      if (!cameraPermission) {
        const granted = await requestCameraPermission();
        if (!granted) {
          Alert.alert('权限不足', '需要相机权限才能拍照');
          return;
        }
      }

      const result = await launchCamera({
        mediaType: 'photo',
        quality,
        maxWidth: 1920,
        maxHeight: 1080,
      });

      if (result.assets && result.assets[0]) {
        handleImageSelection(result.assets[0]);
      }
    } catch (error) {
      Alert.alert('错误', '打开相机失败');
    }
  };

  // 打开相册
  const openGallery = async () => {
    try {
      if (!galleryPermission) {
        const granted = await requestGalleryPermission();
        if (!granted) {
          Alert.alert('权限不足', '需要相册权限才能选择图片');
          return;
        }
      }

      const result = await launchImageLibrary({
        mediaType: 'photo',
        quality,
        selectionLimit: 1,
      });

      if (result.assets && result.assets[0]) {
        handleImageSelection(result.assets[0]);
      }
    } catch (error) {
      Alert.alert('错误', '打开相册失败');
    }
  };

  // 处理图片选择
  const handleImageSelection = async (asset: any) => {
    try {
      // 检查文件大小
      if (asset.fileSize && asset.fileSize > maxSize) {
        Alert.alert(
          '文件过大',
          `图片大小不能超过 ${(maxSize / 1024 / 1024).toFixed(1)}MB`
        );
        return;
      }

      const imageData: ImageData = {
        uri: asset.uri,
        name: asset.fileName || `image_${Date.now()}.jpg`,
        type: asset.type || 'image/jpeg',
        size: asset.fileSize || 0,
        width: asset.width || 0,
        height: asset.height || 0,
      };

      setSelectedImage(asset.uri);
      setShowModal(false);
      onImageSelected(imageData);
    } catch (error) {
      Alert.alert('错误', '图片处理失败');
    }
  };

  return (
    <View style={[styles.container, style]}>
      <TouchableOpacity
        style={styles.button}
        onPress={() => setShowModal(true)}
        activeOpacity={0.7}
      >
        {selectedImage ? (
          <Image source={{ uri: selectedImage }} style={styles.preview} />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>选择图片</Text>
          </View>
        )}
      </TouchableOpacity>

      <Modal
        visible={showModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>选择图片</Text>

            <View style={styles.optionsContainer}>
              {allowCamera && (
                <TouchableOpacity
                  style={styles.optionButton}
                  onPress={openCamera}
                  activeOpacity={0.7}
                >
                  <Text style={styles.optionText}>拍照</Text>
                </TouchableOpacity>
              )}

              {allowGallery && (
                <TouchableOpacity
                  style={styles.optionButton}
                  onPress={openGallery}
                  activeOpacity={0.7}
                >
                  <Text style={styles.optionText}>从相册选择</Text>
                </TouchableOpacity>
              )}
            </View>

            <Button
              title="取消"
              variant="outline"
              onPress={() => setShowModal(false)}
            />
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    // 样式已在主题系统中定义
  },
  button: {
    width: '100%',
    aspectRatio: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    overflow: 'hidden',
  },
  preview: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 300,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 20,
  },
  optionsContainer: {
    gap: 12,
    marginBottom: 20,
  },
  optionButton: {
    padding: 16,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    alignItems: 'center',
  },
  optionText: {
    fontSize: 16,
    fontWeight: '500',
  },
});
```

#### ProcessingProgress 处理进度组件

```typescript
// components/business/ProcessingProgress/ProcessingProgress.tsx
import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  ViewStyle,
} from 'react-native';
import { useWebSocket } from '@hooks/useWebSocket';
import { useTheme } from '@theme';

export interface ProcessingProgressProps {
  taskId: string;
  onCancel?: () => void;
  onComplete?: (result: ProcessingResult) => void;
  onError?: (error: string) => void;
  style?: ViewStyle;
}

interface ProcessingStatus {
  taskId: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number; // 0-100
  estimatedTime?: number; // 预估剩余时间（秒）
  currentStep?: string;
  error?: string;
}

export const ProcessingProgress: React.FC<ProcessingProgressProps> = ({
  taskId,
  onCancel,
  onComplete,
  onError,
  style,
}) => {
  const { theme } = useTheme();
  const progressAnim = useRef(new Animated.Value(0)).current;
  const [status, setStatus] = React.useState<ProcessingStatus>({
    taskId,
    status: 'pending',
    progress: 0,
  });

  // WebSocket连接
  const { lastMessage, sendMessage } = useWebSocket(`/processing/${taskId}`);

  // 监听WebSocket消息
  useEffect(() => {
    if (lastMessage) {
      const message = JSON.parse(lastMessage.data);

      switch (message.type) {
        case 'progress_update':
          updateProgress(message.data);
          break;
        case 'processing_completed':
          handleCompleted(message.data);
          break;
        case 'processing_error':
          handleError(message.data);
          break;
      }
    }
  }, [lastMessage]);

  // 更新进度
  const updateProgress = (data: Partial<ProcessingStatus>) => {
    setStatus(prev => ({ ...prev, ...data }));

    if (data.progress !== undefined) {
      Animated.timing(progressAnim, {
        toValue: data.progress,
        duration: 300,
        useNativeDriver: false,
      }).start();
    }
  };

  // 处理完成
  const handleCompleted = (result: ProcessingResult) => {
    setStatus(prev => ({
      ...prev,
      status: 'completed',
      progress: 100,
    }));

    onComplete?.(result);
  };

  // 处理错误
  const handleError = (errorData: { error: string }) => {
    setStatus(prev => ({
      ...prev,
      status: 'error',
      error: errorData.error,
    }));

    onError?.(errorData.error);
  };

  // 取消处理
  const handleCancel = () => {
    sendMessage(JSON.stringify({
      type: 'cancel_processing',
      taskId,
    }));

    onCancel?.();
  };

  const getStatusText = () => {
    switch (status.status) {
      case 'pending':
        return '准备中...';
      case 'processing':
        return status.currentStep || '处理中...';
      case 'completed':
        return '处理完成';
      case 'error':
        return status.error || '处理失败';
      default:
        return '';
    }
  };

  const getProgressColor = () => {
    switch (status.status) {
      case 'completed':
        return theme.colors.success;
      case 'error':
        return theme.colors.error;
      default:
        return theme.colors.primary;
    }
  };

  return (
    <View style={[styles.container, style]}>
      {/* 进度条 */}
      <View style={styles.progressBar}>
        <View style={styles.progressBackground}>
          <Animated.View
            style={[
              styles.progressFill,
              {
                width: progressAnim.interpolate({
                  inputRange: [0, 100],
                  outputRange: ['0%', '100%'],
                  extrapolate: 'clamp',
                }),
                backgroundColor: getProgressColor(),
              },
            ]}
          />
        </View>
        <Text style={styles.progressText}>
          {Math.round(status.progress)}%
        </Text>
      </View>

      {/* 状态文字 */}
      <Text style={styles.statusText}>
        {getStatusText()}
      </Text>

      {/* 预估时间 */}
      {status.estimatedTime && status.status === 'processing' && (
        <Text style={styles.estimatedTime}>
          预估剩余时间: {Math.ceil(status.estimatedTime)}秒
        </Text>
      )}

      {/* 取消按钮 */}
      {status.status === 'processing' && onCancel && (
        <View style={styles.actions}>
          {/* Button 组件 */}
          <Button
            title="取消处理"
            variant="outline"
            onPress={handleCancel}
          />
        </View>
      )}

      {/* 重试按钮 */}
      {status.status === 'error' && onError && (
        <View style={styles.actions}>
          <Button
            title="重试"
            onPress={() => {
              setStatus(prev => ({ ...prev, status: 'pending', progress: 0 }));
              // 重新发起处理请求
            }}
          />
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 12,
    margin: 16,
  },
  progressBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  progressBackground: {
    flex: 1,
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    marginRight: 12,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    fontWeight: '600',
    minWidth: 40,
    textAlign: 'right',
  },
  statusText: {
    fontSize: 16,
    marginBottom: 4,
  },
  estimatedTime: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  actions: {
    marginTop: 16,
  },
});
```

### 2. 算法选择组件

#### AlgorithmSelector 算法选择器

```typescript
// components/business/AlgorithmSelector/AlgorithmSelector.tsx
import React, { useState, useMemo } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  Image,
  StyleSheet,
  ViewStyle,
  ListRenderItem,
} from 'react-native';
import { useAlgorithmStore } from '@stores/algorithmStore';
import { Card, Input, LoadingIndicator } from '@components/common';
import { useTheme } from '@theme';
import { Algorithm, AlgorithmCategory } from '@types/algorithm';

export interface AlgorithmSelectorProps {
  onAlgorithmSelected: (algorithm: Algorithm) => void;
  selectedAlgorithmId?: string;
  showRecommended?: boolean;
  style?: ViewStyle;
}

export const AlgorithmSelector: React.FC<AlgorithmSelectorProps> = ({
  onAlgorithmSelected,
  selectedAlgorithmId,
  showRecommended = true,
  style,
}) => {
  const { theme } = useTheme();
  const { algorithms, categories, recommendedAlgorithms, loading, searchAlgorithms } = useAlgorithmStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // 过滤算法
  const filteredAlgorithms = useMemo(() => {
    let filtered = algorithms;

    // 按分类过滤
    if (selectedCategory) {
      filtered = filtered.filter(algo => algo.categoryId === selectedCategory);
    }

    // 按搜索关键词过滤
    if (searchQuery) {
      filtered = searchAlgorithms(searchQuery, filtered);
    }

    return filtered;
  }, [algorithms, selectedCategory, searchQuery, searchAlgorithms]);

  // 渲染算法项
  const renderAlgorithm: ListRenderItem<Algorithm> = ({ item }) => {
    const isSelected = item.id === selectedAlgorithmId;
    const isRecommended = recommendedAlgorithms.some(algo => algo.id === item.id);

    return (
      <TouchableOpacity
        style={[
          styles.algorithmItem,
          isSelected && styles.selectedItem,
          { borderColor: isSelected ? theme.colors.primary : theme.colors.border },
        ]}
        onPress={() => onAlgorithmSelected(item)}
        activeOpacity={0.7}
      >
        {/* 算法图标/缩略图 */}
        {item.thumbnail && (
          <Image source={{ uri: item.thumbnail }} style={styles.algorithmImage} />
        )}

        <View style={styles.algorithmInfo}>
          <View style={styles.algorithmHeader}>
            <Text style={styles.algorithmName}>{item.name}</Text>
            {isRecommended && (
              <View style={styles.recommendedBadge}>
                <Text style={styles.recommendedText}>推荐</Text>
              </View>
            )}
          </View>

          <Text style={styles.algorithmDescription} numberOfLines={2}>
            {item.description}
          </Text>

          <View style={styles.algorithmMeta}>
            <Text style={styles.algorithmType}>{item.type}</Text>
            <Text style={styles.algorithmSpeed}>
              速度: {getSpeedText(item.speed)}
            </Text>
          </View>
        </View>
      </TouchableOpacity>
    );
  };

  // 渲染分类标签
  const renderCategoryTabs = () => (
    <View style={styles.categoryTabs}>
      <TouchableOpacity
        style={[
          styles.categoryTab,
          !selectedCategory && styles.activeCategoryTab,
          { backgroundColor: !selectedCategory ? theme.colors.primary : theme.colors.surface },
        ]}
        onPress={() => setSelectedCategory(null)}
      >
        <Text style={[
          styles.categoryTabText,
          !selectedCategory && styles.activeCategoryTabText,
          { color: !selectedCategory ? theme.colors.white : theme.colors.text },
        ]}>
          全部
        </Text>
      </TouchableOpacity>

      {categories.map((category) => (
        <TouchableOpacity
          key={category.id}
          style={[
            styles.categoryTab,
            selectedCategory === category.id && styles.activeCategoryTab,
            {
              backgroundColor: selectedCategory === category.id
                ? theme.colors.primary
                : theme.colors.surface
            },
          ]}
          onPress={() => setSelectedCategory(category.id)}
        >
          <Text style={[
            styles.categoryTabText,
            selectedCategory === category.id && styles.activeCategoryTabText,
            {
              color: selectedCategory === category.id
                ? theme.colors.white
                : theme.colors.text
            },
          ]}>
            {category.name}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  if (loading && algorithms.length === 0) {
    return (
      <View style={[styles.loadingContainer, style]}>
        <LoadingIndicator size="large" />
        <Text style={styles.loadingText}>加载算法中...</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      {/* 推荐算法 */}
      {showRecommended && recommendedAlgorithms.length > 0 && !searchQuery && !selectedCategory && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>推荐算法</Text>
          <FlatList
            data={recommendedAlgorithms}
            renderItem={renderAlgorithm}
            keyExtractor={(item) => item.id}
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.recommendedList}
          />
        </View>
      )}

      {/* 搜索框 */}
      <Input
        placeholder="搜索算法..."
        value={searchQuery}
        onChangeText={setSearchQuery}
        leftIcon={<SearchIcon size={20} color={theme.colors.textSecondary} />}
        containerStyle={styles.searchInput}
      />

      {/* 分类标签 */}
      {renderCategoryTabs()}

      {/* 算法列表 */}
      <FlatList
        data={filteredAlgorithms}
        renderItem={renderAlgorithm}
        keyExtractor={(item) => item.id}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.algorithmList}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>未找到匹配的算法</Text>
          </View>
        }
      />
    </View>
  );
};

// 辅助函数
const getSpeedText = (speed: 'fast' | 'medium' | 'slow'): string => {
  const speedMap = {
    fast: '快',
    medium: '中',
    slow: '慢',
  };
  return speedMap[speed];
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
  },
  recommendedList: {
    paddingRight: 16,
  },
  searchInput: {
    marginBottom: 16,
  },
  categoryTabs: {
    flexDirection: 'row',
    marginBottom: 16,
    gap: 8,
  },
  categoryTab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  activeCategoryTab: {
    // 样式已在动态样式中定义
  },
  categoryTabText: {
    fontSize: 14,
    fontWeight: '500',
  },
  activeCategoryTabText: {
    // 样式已在动态样式中定义
  },
  algorithmList: {
    paddingHorizontal: 16,
  },
  algorithmItem: {
    flexDirection: 'row',
    padding: 16,
    marginBottom: 12,
    borderRadius: 12,
    borderWidth: 1,
    backgroundColor: '#fff',
  },
  selectedItem: {
    // 样式已在动态样式中定义
  },
  algorithmImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 12,
  },
  algorithmInfo: {
    flex: 1,
  },
  algorithmHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  algorithmName: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  recommendedBadge: {
    backgroundColor: '#ff6b6b',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
  },
  recommendedText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '600',
  },
  algorithmDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
    lineHeight: 20,
  },
  algorithmMeta: {
    flexDirection: 'row',
    gap: 16,
  },
  algorithmType: {
    fontSize: 12,
    color: '#999',
  },
  algorithmSpeed: {
    fontSize: 12,
    color: '#999',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
  },
});
```

---

## 🎭 动画组件设计

### 1. 过渡动画组件

#### FadeIn 淡入动画

```typescript
// components/animation/FadeIn/FadeIn.tsx
import React, { useEffect, useRef } from 'react';
import {
  Animated,
  ViewStyle,
  StyleSheet,
} from 'react-native';

export interface FadeInProps {
  children: React.ReactNode;
  duration?: number;
  delay?: number;
  fromValue?: number;
  toValue?: number;
  style?: ViewStyle;
}

export const FadeIn: React.FC<FadeInProps> = ({
  children,
  duration = 300,
  delay = 0,
  fromValue = 0,
  toValue = 1,
  style,
}) => {
  const opacityValue = useRef(new Animated.Value(fromValue)).current;

  useEffect(() => {
    const timer = setTimeout(() => {
      Animated.timing(opacityValue, {
        toValue,
        duration,
        useNativeDriver: true,
      }).start();
    }, delay);

    return () => clearTimeout(timer);
  }, [duration, delay, fromValue, toValue]);

  return (
    <Animated.View style={[{ opacity: opacityValue }, style]}>
      {children}
    </Animated.View>
  );
};
```

#### SlideIn 滑入动画

```typescript
// components/animation/SlideIn/SlideIn.tsx
import React, { useEffect, useRef } from 'react';
import {
  Animated,
  ViewStyle,
  StyleSheet,
} from 'react-native';

export type SlideDirection = 'up' | 'down' | 'left' | 'right';

export interface SlideInProps {
  children: React.ReactNode;
  direction?: SlideDirection;
  distance?: number;
  duration?: number;
  delay?: number;
  style?: ViewStyle;
}

export const SlideIn: React.FC<SlideInProps> = ({
  children,
  direction = 'up',
  distance = 50,
  duration = 300,
  delay = 0,
  style,
}) => {
  const translateValue = useRef(
    new Animated.Value(getInitialValue(direction, distance))
  ).current;

  useEffect(() => {
    const timer = setTimeout(() => {
      Animated.timing(translateValue, {
        toValue: 0,
        duration,
        useNativeDriver: true,
      }).start();
    }, delay);

    return () => clearTimeout(timer);
  }, [direction, distance, duration, delay]);

  const getTransformStyle = () => {
    switch (direction) {
      case 'up':
      case 'down':
        return [{ translateY: translateValue }];
      case 'left':
      case 'right':
        return [{ translateX: translateValue }];
      default:
        return [];
    }
  };

  return (
    <Animated.View style={[{ transform: getTransformStyle() }, style]}>
      {children}
    </Animated.View>
  );
};

function getInitialValue(direction: SlideDirection, distance: number): number {
  switch (direction) {
    case 'up':
      return distance;
    case 'down':
      return -distance;
    case 'left':
      return distance;
    case 'right':
      return -distance;
    default:
      return 0;
  }
}
```

### 2. 手势动画组件

#### SwipeAction 滑动操作

```typescript
// components/gesture/SwipeAction/SwipeAction.tsx
import React, { useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  PanGestureHandler,
  State,
} from 'react-native-gesture-handler';
import { useTheme } from '@theme';

export interface SwipeActionProps {
  children: React.ReactNode;
  leftActions?: SwipeActionItem[];
  rightActions?: SwipeActionItem[];
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  threshold?: number;
}

export interface SwipeActionItem {
  title: string;
  onPress: () => void;
  color?: string;
  backgroundColor?: string;
  width?: number;
}

export const SwipeAction: React.FC<SwipeActionProps> = ({
  children,
  leftActions = [],
  rightActions = [],
  onSwipeLeft,
  onSwipeRight,
  threshold = 100,
}) => {
  const { theme } = useTheme();
  const translateX = useRef(new Animated.Value(0)).current;
  const [isSwiped, setIsSwiped] = useState(false);

  const onGestureEvent = Animated.event(
    [{ nativeEvent: { translationX: translateX } }],
    { useNativeDriver: true }
  );

  const onHandlerStateChange = (event: any) => {
    const { translationX, state } = event.nativeEvent;

    if (state === State.END) {
      const shouldSwipeLeft = translationX > threshold;
      const shouldSwipeRight = translationX < -threshold;

      if (shouldSwipeLeft && leftActions.length > 0) {
        onSwipeLeft?.();
        snapToLeft();
      } else if (shouldSwipeRight && rightActions.length > 0) {
        onSwipeRight?.();
        snapToRight();
      } else {
        snapToCenter();
      }
    }
  };

  const snapToLeft = () => {
    const leftWidth = leftActions.reduce((total, action) =>
      total + (action.width || 80), 0
    );

    Animated.spring(translateX, {
      toValue: leftWidth,
      useNativeDriver: true,
    }).start();

    setIsSwiped(true);
  };

  const snapToRight = () => {
    const rightWidth = rightActions.reduce((total, action) =>
      total + (action.width || 80), 0
    );

    Animated.spring(translateX, {
      toValue: -rightWidth,
      useNativeDriver: true,
    }).start();

    setIsSwiped(true);
  };

  const snapToCenter = () => {
    Animated.spring(translateX, {
      toValue: 0,
      useNativeDriver: true,
    }).start();

    setIsSwiped(false);
  };

  const renderLeftActions = () => {
    if (leftActions.length === 0) return null;

    const totalWidth = leftActions.reduce((total, action) =>
      total + (action.width || 80), 0
    );

    return (
      <View style={[styles.actionsContainer, { right: 0, width: totalWidth }]}>
        {leftActions.map((action, index) => (
          <TouchableOpacity
            key={`left-${index}`}
            style={[
              styles.actionButton,
              {
                backgroundColor: action.backgroundColor || theme.colors.primary,
                width: action.width || 80,
                right: totalWidth - (index + 1) * (action.width || 80),
              },
            ]}
            onPress={() => {
              action.onPress();
              snapToCenter();
            }}
          >
            <Text style={[
              styles.actionText,
              { color: action.color || theme.colors.white }
            ]}>
              {action.title}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const renderRightActions = () => {
    if (rightActions.length === 0) return null;

    const totalWidth = rightActions.reduce((total, action) =>
      total + (action.width || 80), 0
    );

    return (
      <View style={[styles.actionsContainer, { left: 0, width: totalWidth }]}>
        {rightActions.map((action, index) => (
          <TouchableOpacity
            key={`right-${index}`}
            style={[
              styles.actionButton,
              {
                backgroundColor: action.backgroundColor || theme.colors.error,
                width: action.width || 80,
                left: index * (action.width || 80),
              },
            ]}
            onPress={() => {
              action.onPress();
              snapToCenter();
            }}
          >
            <Text style={[
              styles.actionText,
              { color: action.color || theme.colors.white }
            ]}>
              {action.title}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {renderLeftActions()}
      {renderRightActions()}

      <PanGestureHandler
        onGestureEvent={onGestureEvent}
        onHandlerStateChange={onHandlerStateChange}
      >
        <Animated.View
          style={[
            styles.content,
            {
              transform: [{ translateX }],
            },
          ]}
        >
          {children}
        </Animated.View>
      </PanGestureHandler>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  content: {
    backgroundColor: '#fff',
  },
  actionsContainer: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    flexDirection: 'row',
  },
  actionButton: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  actionText: {
    fontSize: 14,
    fontWeight: '600',
  },
});
```

---

## 📚 组件最佳实践

### 1. 性能优化原则

#### React.memo使用

```typescript
// ✅ 使用React.memo优化纯组件
export const ImageCard = React.memo<ImageCardProps>(({ image, onPress }) => {
  return (
    <TouchableOpacity onPress={() => onPress(image)}>
      <Image source={{ uri: image.uri }} />
      <Text>{image.title}</Text>
    </TouchableOpacity>
  );
}, (prevProps, nextProps) => {
  // 自定义比较函数
  return prevProps.image.id === nextProps.image.id;
});
```

#### useMemo和useCallback使用

```typescript
// ✅ 优化渲染性能
const AlgorithmList: React.FC = ({ algorithms, onSelect }) => {
  const renderAlgorithm = useCallback((algorithm: Algorithm) => (
    <AlgorithmItem
      key={algorithm.id}
      algorithm={algorithm}
      onPress={() => onSelect(algorithm)}
    />
  ), [onSelect]);

  const sortedAlgorithms = useMemo(() => {
    return algorithms.sort((a, b) => a.name.localeCompare(b.name));
  }, [algorithms]);

  return (
    <FlatList
      data={sortedAlgorithms}
      renderItem={({ item }) => renderAlgorithm(item)}
      keyExtractor={(item) => item.id}
    />
  );
};
```

### 2. 可访问性设计

#### 无障碍属性

```typescript
// ✅ 完善的无障碍支持
const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  accessibilityLabel,
  accessibilityHint,
  ...props
}) => {
  return (
    <TouchableOpacity
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel || title}
      accessibilityHint={accessibilityHint || `点击${title}`}
      accessibilityState={{ disabled: props.disabled }}
      {...props}
    >
      <Text>{title}</Text>
    </TouchableOpacity>
  );
};
```

### 3. 测试友好的组件设计

#### 测试ID和属性

```typescript
// ✅ 测试友好的组件
export const ImagePicker: React.FC<ImagePickerProps> = ({ onImageSelected }) => {
  return (
    <View testID="image-picker">
      <TouchableOpacity
        testID="camera-button"
        accessibilityLabel="打开相机"
        onPress={handleCameraPress}
      >
        <Text>拍照</Text>
      </TouchableOpacity>

      <TouchableOpacity
        testID="gallery-button"
        accessibilityLabel="从相册选择"
        onPress={handleGalleryPress}
      >
        <Text>相册</Text>
      </TouchableOpacity>
    </View>
  );
};
```

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**下次更新**: 根据组件开发进度和需求变化持续更新
