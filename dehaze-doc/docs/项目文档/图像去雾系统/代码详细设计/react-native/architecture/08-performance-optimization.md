# 性能优化策略

**文档版本**: v1.0
**最后更新**: 2025-11-22
**项目名称**: dehaze-react-native
**目标平台**: iOS、Android

---

## 📋 文档概述

本文档详细描述了Dehaze React Native应用的性能优化策略，包括启动性能、渲染性能、内存管理、网络优化、图片处理优化和电池使用优化。基于移动端特性，提供全面的性能监控和优化方案，确保应用在各种设备上都能提供流畅的用户体验。

---

## 🎯 性能优化目标

### 核心性能指标

#### 1. 启动性能
- **应用启动时间**: 冷启动 < 3秒，热启动 < 1秒
- **首屏渲染时间**: 从启动到首屏显示 < 2秒
- **交互响应时间**: 用户操作到界面响应 < 100ms
- **页面切换时间**: 页面间切换动画 < 300ms

#### 2. 运行时性能
- **帧率**: 保持60FPS，最低不低于30FPS
- **CPU使用率**: 正常使用时 < 30%，峰值 < 70%
- **内存使用**: 低配设备 < 150MB，高配设备 < 300MB
- **电池消耗**: 连续使用1小时电量消耗 < 15%

#### 3. 网络性能
- **API响应时间**: 简单接口 < 200ms，复杂接口 < 1000ms
- **图片加载时间**: 缩略图 < 500ms，高清图 < 2000ms
- **离线体验**: 核心功能支持离线使用
- **数据传输**: 启用gzip压缩，减少50%+传输量

---

## 🚀 启动性能优化

### 1. 启动流程优化

```mermaid
sequenceDiagram
    participant App as 应用启动
    participant Bridge as Bridge初始化
    participant Bundle as Bundle加载
    component Component as 首屏组件
    participant Data as 数据加载
    participant Render as 首屏渲染

    App->>Bridge: 初始化JS Bridge
    Bridge->>Bundle: 加载JS Bundle
    Bundle->>Component: 渲染首屏组件
    Component->>Data: 异步加载必要数据
    Data->>Component: 返回数据
    Component->>Render: 完成首屏渲染
```

### 2. 启动优化实现

```typescript
// 启动性能管理器
class StartupPerformanceManager {
  private static instance: StartupPerformanceManager;
  private startupMetrics: StartupMetrics = {};
  private startTimes: Record<string, number> = {};

  static getInstance(): StartupPerformanceManager {
    if (!StartupPerformanceManager.instance) {
      StartupPerformanceManager.instance = new StartupPerformanceManager();
    }
    return StartupPerformanceManager.instance;
  }

  // 记录启动时间点
  markStartTime(marker: string) {
    this.startTimes[marker] = Date.now();
  }

  // 计算启动时间间隔
  measureInterval(fromMarker: string, toMarker: string): number {
    const from = this.startTimes[fromMarker];
    const to = this.startTimes[toMarker];
    return from && to ? to - from : 0;
  }

  // 获取启动性能报告
  getStartupReport(): StartupMetrics {
    return {
      totalTime: this.measureInterval('app_start', 'first_render'),
      bridgeTime: this.measureInterval('app_start', 'bridge_ready'),
      bundleTime: this.measureInterval('bridge_ready', 'bundle_loaded'),
      renderTime: this.measureInterval('bundle_loaded', 'first_render'),
      dataLoadTime: this.measureInterval('first_render', 'data_loaded'),
    };
  }
}

// 启动优化App组件
const OptimizedApp = () => {
  const startupManager = StartupPerformanceManager.getInstance();

  useEffect(() => {
    // 记录应用启动时间
    startupManager.markStartTime('app_start');
  }, []);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <AppProvider>
          <PerformanceMonitor>
            <PreloadResources>
              <AppNavigation />
            </PreloadResources>
          </PerformanceMonitor>
        </AppProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
};

// 资源预加载组件
const PreloadResources: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isPreloaded, setIsPreloaded] = useState(false);
  const startupManager = StartupPerformanceManager.getInstance();

  useEffect(() => {
    const preloadResources = async () => {
      try {
        startupManager.markStartTime('bridge_ready');

        // 并行预加载关键资源
        await Promise.all([
          preloadFonts(),
          preloadIcons(),
          preloadCriticalData(),
          preloadUserPreferences(),
        ]);

        startupManager.markStartTime('bundle_loaded');
        setIsPreloaded(true);
      } catch (error) {
        console.error('Resource preloading failed:', error);
        // 即使预加载失败也要继续启动
        setIsPreloaded(true);
      }
    };

    preloadResources();
  }, []);

  if (!isPreloaded) {
    return <StartupScreen />;
  }

  return <>{children}</>;
};

// 启动屏幕组件
const StartupScreen = () => {
  const [loadingText, setLoadingText] = useState('正在启动...');

  useEffect(() => {
    const loadingTexts = [
      '正在初始化应用...',
      '正在加载资源...',
      '正在准备界面...',
    ];

    let index = 0;
    const interval = setInterval(() => {
      index = (index + 1) % loadingTexts.length;
      setLoadingText(loadingTexts[index]);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <View style={styles.startupContainer}>
      <Image
        source={require('../../assets/images/logo.png')}
        style={styles.logo}
        resizeMode="contain"
      />
      <Text style={styles.title}>Dehaze System</Text>
      <Text style={styles.subtitle}>专业图像去雾系统</Text>
      <ActivityIndicator
        size="large"
        color="#3B82F6"
        style={styles.loadingIndicator}
      />
      <Text style={styles.loadingText}>{loadingText}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  startupContainer: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: 120,
    height: 120,
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#6B7280',
    marginBottom: 48,
  },
  loadingIndicator: {
    marginBottom: 16,
  },
  loadingText: {
    fontSize: 14,
    color: '#9CA3AF',
  },
});
```

### 3. 懒加载与代码分割

```typescript
// 路由级别的懒加载
const LazyHomeScreen = lazy(() => import('../pages/home/HomeScreen'));
const LazyImageInputScreen = lazy(() => import('../pages/imageInput/ImageInputScreen'));
const LazyAlgorithmSelectScreen = lazy(() => import('../pages/algorithmSelect/AlgorithmSelectScreen'));
const LazyDehazeProcessingScreen = lazy(() => import('../pages/dehazeProcessing/DehazeProcessingScreen'));
const LazyEffectComparisonScreen = lazy(() => import('../pages/effectComparison/EffectComparisonScreen'));

// 懒加载导航器
const LazyTabNavigator = () => {
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    const unsubscribe = useFocusEffect(
      useCallback(() => {
        setIsFocused(true);
        return () => setIsFocused(false);
      }, [])
    );

    return unsubscribe;
  }, []);

  if (!isFocused) {
    return <View style={{ flex: 1, backgroundColor: '#F3F4F6' }} />;
  }

  return (
    <Suspense fallback={<LoadingScreen />}>
      <Tab.Navigator>
        <Tab.Screen
          name="Home"
          component={LazyHomeScreen}
          options={{
            title: '首页',
            tabBarIcon: ({ color, size }) => (
              <Icon name="home" size={size} color={color} />
            ),
          }}
        />
        {/* 其他标签页... */}
      </Tab.Navigator>
    </Suspense>
  );
};

// 组件级别的懒加载
const LazyImage = React.memo<{
  source: ImageSourcePropType;
  style?: ImageStyle;
  placeholder?: React.ReactNode;
}>(({ source, style, placeholder }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const viewRef = useRef<View>(null);

  useEffect(() => {
    const observer = viewRef.current ?
      viewRef.current.measureLayout.bind(viewRef.current) : null;

    // 使用IntersectionObserver检测图片是否在视口内
    const subscription = InteractionManager.runAfterInteractions(() => {
      requestAnimationFrame(() => {
        setIsVisible(true);
      });
    });

    return () => subscription.cancel();
  }, []);

  const handleLoad = useCallback(() => {
    setIsLoaded(true);
  }, []);

  if (!isVisible) {
    return (
      <View ref={viewRef} style={style}>
        {placeholder || <View style={styles.imagePlaceholder} />}
      </View>
    );
  }

  return (
    <View style={style}>
      {!isLoaded && placeholder && placeholder}
      <Image
        source={source}
        style={[
          StyleSheet.absoluteFillObject,
          { opacity: isLoaded ? 1 : 0 }
        ]}
        onLoad={handleLoad}
      />
    </View>
  );
});
```

---

## 🎨 渲染性能优化

### 1. 组件渲染优化

```typescript
// 性能优化的图像列表组件
const OptimizedImageList = React.memo<{
  images: ImageInfo[];
  onImagePress: (image: ImageInfo) => void;
}>(({ images, onImagePress }) => {
  const { isTablet } = useResponsive();

  // 获取项目布局
  const getItemLayout = useCallback((data: any, index: number) => ({
    length: isTablet ? 200 : 150,
    offset: (isTablet ? 200 : 150) * index,
    index,
  }), [isTablet]);

  // 获取项目key
  const keyExtractor = useCallback((item: ImageInfo) => item.imageId, []);

  // 渲染项目
  const renderItem = useCallback(({ item }: { item: ImageInfo }) => (
    <MemoizedImageCard
      image={item}
      onPress={onImagePress}
      width={isTablet ? 180 : 140}
      height={isTablet ? 180 : 140}
    />
  ), [onImagePress, isTablet]);

  return (
    <FlatList
      data={images}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      getItemLayout={getItemLayout}
      numColumns={isTablet ? 3 : 2}
      removeClippedSubviews={true}
      maxToRenderPerBatch={isTablet ? 12 : 8}
      updateCellsBatchingPeriod={50}
      initialNumToRender={isTablet ? 12 : 8}
      windowSize={isTablet ? 21 : 15}
      showsVerticalScrollIndicator={false}
      contentContainerStyle={{
        paddingHorizontal: isTablet ? 16 : 12,
        paddingVertical: isTablet ? 20 : 16,
      }}
    />
  );
});

// 记忆化图像卡片
const MemoizedImageCard = React.memo<{
  image: ImageInfo;
  onPress: (image: ImageInfo) => void;
  width: number;
  height: number;
}>(({ image, onPress, width, height }) => {
  const cardStyle = useMemo(() => ({
    width,
    height,
    margin: 4,
    borderRadius: 12,
    backgroundColor: '#FFFFFF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  }), [width, height]);

  const imageStyle = useMemo(() => ({
    width: '100%',
    height: width * 0.75,
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
  }), [width]);

  return (
    <TouchableOpacity style={cardStyle} onPress={() => onPress(image)}>
      <OptimizedImage
        source={{ uri: image.thumbnailUrl }}
        style={imageStyle}
        resizeMode="cover"
      />
      <View style={styles.cardContent}>
        <Text style={styles.imageTitle} numberOfLines={1}>
          {image.filename}
        </Text>
        <Text style={styles.imageInfo}>
          {image.metadata.width} × {image.metadata.height}
        </Text>
      </View>
    </TouchableOpacity>
  );
});

// 高性能图像组件
const OptimizedImage = React.memo<{
  source: ImageSourcePropType;
  style?: ImageStyle;
  resizeMode?: ResizeMode;
  onLoad?: () => void;
}>(({ source, style, resizeMode = 'cover', onLoad }) => {
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const imageRef = useRef<Image>(null);

  // 图片加载成功处理
  const handleLoad = useCallback((event: ImageLoadSuccessEvent) => {
    const { width, height } = event.nativeEvent.source;
    setImageSize({ width, height });
    onLoad?.();
  }, [onLoad]);

  // 图片尺寸计算
  const computedStyle = useMemo(() => {
    if (!imageSize || !style) return style;

    const { width: styleWidth, height: styleHeight } = StyleSheet.flatten(style);

    if (typeof styleWidth === 'number' && typeof styleHeight === 'number') {
      const targetRatio = styleWidth / styleHeight;
      const imageRatio = imageSize.width / imageSize.height;

      if (imageRatio > targetRatio) {
        return { ...style, width: styleWidth };
      } else {
        return { ...style, height: styleHeight };
      }
    }

    return style;
  }, [imageSize, style]);

  return (
    <Image
      ref={imageRef}
      source={source}
      style={computedStyle}
      resizeMode={resizeMode}
      onLoad={handleLoad}
      blurRadius={1}
    />
  );
});
```

### 2. 动画性能优化

```typescript
// 高性能动画管理器
class AnimationManager {
  private static instance: AnimationManager;
  private activeAnimations: Map<string, Animated.Value> = new Map();
  private animationQueue: Array<() => void> = [];
  private isProcessingQueue = false;

  static getInstance(): AnimationManager {
    if (!AnimationManager.instance) {
      AnimationManager.instance = new AnimationManager();
    }
    return AnimationManager.instance;
  }

  // 创建驱动动画
  createValue(initialValue: number, config?: Animated.ValueAnimationConfig): Animated.Value {
    const value = new Animated.Value(initialValue);

    if (config) {
      this.activeAnimations.set(`animation_${Date.now()}_${Math.random()}`, value);
    }

    return value;
  }

  // 批量执行动画
  queueAnimation(animationFn: () => void) {
    this.animationQueue.push(animationFn);
    this.processQueue();
  }

  private processQueue() {
    if (this.isProcessingQueue || this.animationQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    InteractionManager.runAfterInteractions(() => {
      const animationFn = this.animationQueue.shift();
      if (animationFn) {
        animationFn();
      }

      this.isProcessingQueue = false;

      if (this.animationQueue.length > 0) {
        requestAnimationFrame(() => this.processQueue());
      }
    });
  }

  // 清理所有动画
  cleanup() {
    this.activeAnimations.forEach(value => {
      value.stopAnimation();
    });
    this.activeAnimations.clear();
    this.animationQueue = [];
  }
}

// 高性能页面过渡动画
const OptimizedPageTransition = ({
  isVisible,
  children,
  animationType = 'fade',
  duration = 300,
}: {
  isVisible: boolean;
  children: React.ReactNode;
  animationType?: 'fade' | 'slide' | 'scale';
  duration?: number;
}) => {
  const animationManager = AnimationManager.getInstance();
  const animatedValue = animationManager.createValue(0);

  useEffect(() => {
    animationManager.queueAnimation(() => {
      if (isVisible) {
        switch (animationType) {
          case 'fade':
            Animated.timing(animatedValue, {
              toValue: 1,
              duration,
              useNativeDriver: true,
            }).start();
            break;
          case 'slide':
            Animated.timing(animatedValue, {
              toValue: 1,
              duration,
              useNativeDriver: true,
            }).start();
            break;
          case 'scale':
            Animated.spring(animatedValue, {
              toValue: 1,
              tension: 100,
              friction: 8,
              useNativeDriver: true,
            }).start();
            break;
        }
      } else {
        Animated.timing(animatedValue, {
          toValue: 0,
          duration: duration / 2,
          useNativeDriver: true,
        }).start();
      }
    });
  }, [isVisible, animationType, duration, animatedValue]);

  const getAnimationStyle = () => {
    switch (animationType) {
      case 'fade':
        return { opacity: animatedValue };
      case 'slide':
        return {
          transform: [
            {
              translateX: animatedValue.interpolate({
                inputRange: [0, 1],
                outputRange: [300, 0],
              }),
            },
          ],
        };
      case 'scale':
        return {
          transform: [
            {
              scale: animatedValue,
            },
          ],
        };
      default:
        return { opacity: animatedValue };
    }
  };

  if (!isVisible && animatedValue._value === 0) {
    return null;
  }

  return (
    <Animated.View style={getAnimationStyle()}>
      {children}
    </Animated.View>
  );
};

// 性能优化的手势处理
const OptimizedGestureHandler = ({
  children,
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
}: {
  children: React.ReactNode;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
}) => {
  const panGesture = Gesture.Pan()
    .onUpdate((event) => {
      // 限制更新频率
      'worklet';
      const { translationX, translationY } = event;

      // 使用更少的更新来提高性能
      const absX = Math.abs(translationX);
      const absY = Math.abs(translationY);

      if (absX > absY && absX > 50) {
        if (translationX > 0 && onSwipeRight) {
          runOnJS(onSwipeRight)();
        } else if (translationX < 0 && onSwipeLeft) {
          runOnJS(onSwipeLeft)();
        }
      } else if (absY > absX && absY > 50) {
        if (translationY > 0 && onSwipeDown) {
          runOnJS(onSwipeDown)();
        } else if (translationY < 0 && onSwipeUp) {
          runOnJS(onSwipeUp)();
        }
      }
    })
    .onEnd(() => {
      'worklet';
    });

  return (
    <GestureDetector gesture={panGesture}>
      {children}
    </GestureDetector>
  );
};
```

---

## 💾 内存管理优化

### 1. 内存监控管理

```typescript
// 内存监控管理器
class MemoryMonitor {
  private static instance: MemoryMonitor;
  private memoryStats: MemoryStats[] = [];
  private maxHistorySize = 100;
  private cleanupCallbacks: Array<() => void> = [];

  static getInstance(): MemoryMonitor {
    if (!MemoryMonitor.instance) {
      MemoryMonitor.instance = new MemoryMonitor();
    }
    return MemoryMonitor.instance;
  }

  // 记录内存状态
  recordMemoryStats() {
    if (Platform.OS === 'android') {
      // Android内存监控
      import('react-native').then(({ NativeModules }) => {
        if (NativeModules.MemoryMonitor) {
          NativeModules.MemoryMonitor.getMemoryInfo((info: any) => {
            this.addStats({
              timestamp: Date.now(),
              usedMemory: info.usedMemory,
              totalMemory: info.totalMemory,
              percentage: (info.usedMemory / info.totalMemory) * 100,
            });
          });
        }
      });
    } else if (Platform.OS === 'ios') {
      // iOS内存监控（需要原生模块支持）
      this.addStats({
        timestamp: Date.now(),
        usedMemory: 0, // 需要原生模块提供
        totalMemory: 0,
        percentage: 0,
      });
    }
  }

  // 添加内存统计
  private addStats(stats: MemoryStats) {
    this.memoryStats.push(stats);

    // 限制历史记录大小
    if (this.memoryStats.length > this.maxHistorySize) {
      this.memoryStats.shift();
    }

    // 检查内存使用是否过高
    if (stats.percentage > 80) {
      this.triggerMemoryCleanup();
    }
  }

  // 触发内存清理
  triggerMemoryCleanup() {
    console.warn('High memory usage detected, triggering cleanup');

    this.cleanupCallbacks.forEach(callback => {
      try {
        callback();
      } catch (error) {
        console.error('Memory cleanup callback failed:', error);
      }
    });

    // 清理图片缓存
    ImageCache.getInstance().cleanup();

    // 清理未使用的数据
    DataCache.getInstance().cleanup();
  }

  // 注册清理回调
  registerCleanupCallback(callback: () => void) {
    this.cleanupCallbacks.push(callback);
  }

  // 获取内存统计
  getMemoryStats(): MemoryStats[] {
    return [...this.memoryStats];
  }

  // 获取当前内存使用
  getCurrentMemoryUsage(): MemoryStats | null {
    return this.memoryStats[this.memoryStats.length - 1] || null;
  }
}

// 图片缓存管理器
class ImageCache {
  private static instance: ImageCache;
  private cache: Map<string, any> = new Map();
  private maxCacheSize = 50; // 最大缓存数量
  private maxCacheMemory = 100 * 1024 * 1024; // 100MB

  static getInstance(): ImageCache {
    if (!ImageCache.instance) {
      ImageCache.instance = new ImageCache();
    }
    return ImageCache.instance;
  }

  // 获取缓存图片
  get(uri: string): any | null {
    return this.cache.get(uri) || null;
  }

  // 设置缓存图片
  set(uri: string, data: any) {
    // 检查缓存大小
    if (this.cache.size >= this.maxCacheSize) {
      this.evictLeastRecentlyUsed();
    }

    this.cache.set(uri, {
      data,
      lastAccessed: Date.now(),
      size: this.estimateSize(data),
    });
  }

  // 估算数据大小
  private estimateSize(data: any): number {
    try {
      return JSON.stringify(data).length * 2; // 粗略估算
    } catch {
      return 1024; // 默认1KB
    }
  }

  // 淘汰最少使用的图片
  private evictLeastRecentlyUsed() {
    let oldestKey = '';
    let oldestTime = Date.now();

    this.cache.forEach((value, key) => {
      if (value.lastAccessed < oldestTime) {
        oldestTime = value.lastAccessed;
        oldestKey = key;
      }
    });

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  // 清理缓存
  cleanup() {
    const currentTime = Date.now();
    const maxAge = 30 * 60 * 1000; // 30分钟

    this.cache.forEach((value, key) => {
      if (currentTime - value.lastAccessed > maxAge) {
        this.cache.delete(key);
      }
    });
  }

  // 清空缓存
  clear() {
    this.cache.clear();
  }

  // 获取缓存统计
  getStats() {
    let totalSize = 0;
    this.cache.forEach(value => {
      totalSize += value.size;
    });

    return {
      count: this.cache.size,
      totalSize,
      maxSize: this.maxCacheMemory,
    };
  }
}
```

### 2. 组件内存优化

```typescript
// 内存优化的图片查看器
const MemoryOptimizedImageViewer = ({
  images,
  initialIndex = 0,
  onClose,
}: {
  images: string[];
  initialIndex?: number;
  onClose: () => void;
}) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [loadedImages, setLoadedImages] = useState<Set<number>>(new Set([initialIndex]));
  const flatListRef = useRef<FlatList>(null);

  // 预加载相邻图片
  const preloadAdjacentImages = useCallback((index: number) => {
    const adjacentIndices = [index - 1, index + 1].filter(i =>
      i >= 0 && i < images.length && !loadedImages.has(i)
    );

    adjacentIndices.forEach(i => {
      Image.prefetch(images[i]).then(() => {
        setLoadedImages(prev => new Set([...prev, i]));
      });
    });
  }, [images, loadedImages]);

  // 当前索引变化处理
  useEffect(() => {
    preloadAdjacentImages(currentIndex);
  }, [currentIndex, preloadAdjacentImages]);

  // 渲染图片项
  const renderItem = useCallback(({ item, index }: { item: string; index: number }) => (
    <View style={styles.imageContainer}>
      {loadedImages.has(index) ? (
        <Image
          source={{ uri: item }}
          style={styles.fullImage}
          resizeMode="contain"
        />
      ) : (
        <View style={styles.imagePlaceholder}>
          <ActivityIndicator size="large" color="#3B82F6" />
        </View>
      )}
    </View>
  ), [loadedImages]);

  // 获取项目布局
  const getItemLayout = useCallback((data: any, index: number) => ({
    length: Dimensions.get('window').width,
    offset: Dimensions.get('window').width * index,
    index,
  }), []);

  // 内存清理
  useEffect(() => {
    const memoryMonitor = MemoryMonitor.getInstance();

    const cleanupCallback = () => {
      // 保留当前和相邻的图片，清理其他图片
      const keepIndices = new Set([
        currentIndex - 1,
        currentIndex,
        currentIndex + 1,
      ]);

      setLoadedImages(prev => {
        const newSet = new Set<number>();
        prev.forEach(index => {
          if (keepIndices.has(index)) {
            newSet.add(index);
          }
        });
        return newSet;
      });
    };

    memoryMonitor.registerCleanupCallback(cleanupCallback);

    return () => {
      memoryMonitor.cleanup();
    };
  }, [currentIndex]);

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatListRef}
        data={images}
        renderItem={renderItem}
        keyExtractor={(item, index) => `${item}_${index}`}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        initialNumToRender={3}
        maxToRenderPerBatch={3}
        windowSize={5}
        getItemLayout={getItemLayout}
        initialScrollIndex={initialIndex}
        onMomentumScrollEnd={(event) => {
          const index = Math.round(
            event.nativeEvent.contentOffset.x / Dimensions.get('window').width
          );
          setCurrentIndex(index);
        }}
      />
      <TouchableOpacity style={styles.closeButton} onPress={onClose}>
        <Icon name="close" size={24} color="#FFFFFF" />
      </TouchableOpacity>
    </View>
  );
};

// 内存优化的数据列表
const MemoryOptimizedDataList = <T extends { id: string }>({
  data,
  renderItem,
  keyExtractor,
}: {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactElement;
  keyExtractor: (item: T, index: number) => string;
}) => {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 10 });
  const viewabilityConfig = {
    viewAreaCoveragePercentThreshold: 10,
    minimumViewTime: 300,
  };

  const onViewableItemsChanged = useCallback(({ changed, viewableItems }) => {
    if (changed.length > 0) {
      const visibleIndices = viewableItems.map(item => item.index || 0);
      const min = Math.min(...visibleIndices);
      const max = Math.max(...visibleIndices);

      setVisibleRange({
        start: Math.max(0, min - 2),
        end: Math.min(data.length - 1, max + 2),
      });
    }
  }, [data.length]);

  // 过滤可见范围内的数据
  const visibleData = useMemo(() =>
    data.slice(visibleRange.start, visibleRange.end + 1),
    [data, visibleRange]
  );

  // 调整key以匹配原始数据索引
  const adjustedKeyExtractor = useCallback((item: T, index: number) => {
    return keyExtractor(item, visibleRange.start + index);
  }, [keyExtractor, visibleRange.start]);

  const adjustedRenderItem = useCallback(({ item, index }: { item: T; index: number }) => {
    return renderItem(item, visibleRange.start + index);
  }, [renderItem, visibleRange.start]);

  return (
    <FlatList
      data={visibleData}
      renderItem={adjustedRenderItem}
      keyExtractor={adjustedKeyExtractor}
      onViewableItemsChanged={onViewableItemsChanged}
      viewabilityConfig={viewabilityConfig}
      getItemLayout={(data, index) => ({
        length: 100,
        offset: 100 * (visibleRange.start + index),
        index: visibleRange.start + index,
      })}
      initialNumToRender={5}
      maxToRenderPerBatch={5}
      windowSize={10}
    />
  );
};
```

---

## 🌐 网络性能优化

### 1. 请求优化策略

```typescript
// 网络性能管理器
class NetworkPerformanceManager {
  private static instance: NetworkPerformanceManager;
  private requestMetrics: RequestMetric[] = [];
  private requestCache: Map<string, CachedResponse> = new Map();
  private ongoingRequests: Map<string, Promise<any>> = new Map();

  static getInstance(): NetworkPerformanceManager {
    if (!NetworkPerformanceManager.instance) {
      NetworkPerformanceManager.instance = new NetworkPerformanceManager();
    }
    return NetworkPerformanceManager.instance;
  }

  // 记录请求性能
  recordRequestMetric(metric: RequestMetric) {
    this.requestMetrics.push(metric);

    // 限制历史记录
    if (this.requestMetrics.length > 1000) {
      this.requestMetrics.shift();
    }

    // 分析性能
    this.analyzePerformance();
  }

  // 分析性能
  private analyzePerformance() {
    const recentMetrics = this.requestMetrics.slice(-50);
    const avgDuration = recentMetrics.reduce((sum, m) => sum + m.duration, 0) / recentMetrics.length;
    const errorRate = recentMetrics.filter(m => !m.success).length / recentMetrics.length;

    if (avgDuration > 1000) {
      console.warn('Slow network performance detected:', avgDuration);
    }

    if (errorRate > 0.1) {
      console.warn('High error rate detected:', errorRate);
    }
  }

  // 智能缓存请求
  async cachedRequest<T>(
    key: string,
    requestFn: () => Promise<T>,
    ttl: number = 300000 // 5分钟
  ): Promise<T> {
    // 检查缓存
    const cached = this.requestCache.get(key);
    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.data;
    }

    // 检查是否已有进行中的请求
    const ongoing = this.ongoingRequests.get(key);
    if (ongoing) {
      return ongoing;
    }

    // 发起新请求
    const requestPromise = this.performRequest(key, requestFn);
    this.ongoingRequests.set(key, requestPromise);

    try {
      const result = await requestPromise;

      // 缓存结果
      this.requestCache.set(key, {
        data: result,
        timestamp: Date.now(),
      });

      return result;
    } finally {
      this.ongoingRequests.delete(key);
    }
  }

  // 执行请求
  private async performRequest<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    const startTime = Date.now();
    let success = false;
    let responseSize = 0;

    try {
      const result = await requestFn();
      success = true;

      // 估算响应大小
      responseSize = JSON.stringify(result).length;

      return result;
    } catch (error) {
      success = false;
      throw error;
    } finally {
      const duration = Date.now() - startTime;

      this.recordRequestMetric({
        key,
        duration,
        success,
        responseSize,
        timestamp: startTime,
      });
    }
  }

  // 清理缓存
  cleanupCache() {
    const currentTime = Date.now();
    const maxAge = 600000; // 10分钟

    this.requestCache.forEach((value, key) => {
      if (currentTime - value.timestamp > maxAge) {
        this.requestCache.delete(key);
      }
    });
  }

  // 获取性能报告
  getPerformanceReport(): NetworkPerformanceReport {
    const recentMetrics = this.requestMetrics.slice(-100);

    const avgDuration = recentMetrics.reduce((sum, m) => sum + m.duration, 0) / recentMetrics.length;
    const errorRate = recentMetrics.filter(m => !m.success).length / recentMetrics.length;
    const cacheHitRate = this.requestCache.size / (this.requestCache.size + this.ongoingRequests.size);

    return {
      avgDuration,
      errorRate,
      cacheHitRate,
      totalRequests: this.requestMetrics.length,
      cacheSize: this.requestCache.size,
    };
  }
}

// 优化的API客户端
const OptimizedApiClient = {
  // 智能GET请求
  async get<T>(
    url: string,
    params?: any,
    options?: {
      cache?: boolean;
      ttl?: number;
      retry?: boolean;
    }
  ): Promise<T> {
    const cacheKey = `${url}_${JSON.stringify(params || {})}`;
    const networkManager = NetworkPerformanceManager.getInstance();

    const requestFn = async () => {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Accept-Encoding': 'gzip, deflate, br',
          'User-Agent': 'DehazeApp/1.0',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.json();
    };

    if (options?.cache) {
      return networkManager.cachedRequest(cacheKey, requestFn, options.ttl);
    } else {
      return requestFn();
    }
  },

  // 优化的POST请求
  async post<T>(
    url: string,
    data: any,
    options?: {
      timeout?: number;
      retry?: boolean;
    }
  ): Promise<T> {
    const networkManager = NetworkPerformanceManager.getInstance();

    const requestFn = async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), options?.timeout || 10000);

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept-Encoding': 'gzip, deflate, br',
          },
          body: JSON.stringify(data),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      } catch (error) {
        clearTimeout(timeoutId);
        throw error;
      }
    };

    return requestFn();
  },

  // 批量请求
  async batch<T>(
    requests: Array<{ url: string; method?: string; data?: any }>
  ): Promise<T[]> {
    const networkManager = NetworkPerformanceManager.getInstance();

    const batchRequest = async () => {
      const promises = requests.map(async (req) => {
        const response = await fetch(req.url, {
          method: req.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Accept-Encoding': 'gzip, deflate, br',
          },
          body: req.data ? JSON.stringify(req.data) : undefined,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      });

      return Promise.all(promises);
    };

    return networkManager.cachedRequest(
      `batch_${JSON.stringify(requests)}`,
      batchRequest,
      60000 // 1分钟缓存
    );
  },
};
```

### 2. 图片下载优化

```typescript
// 图片下载管理器
class ImageDownloadManager {
  private static instance: ImageDownloadManager;
  private downloadQueue: Map<string, Promise<any>> = new Map();
  private downloadCache: Map<string, CachedImage> = new Map();

  static getInstance(): ImageDownloadManager {
    if (!ImageDownloadManager.instance) {
      ImageDownloadManager.instance = new ImageDownloadManager();
    }
    return ImageDownloadManager.instance;
  }

  // 下载图片（支持多尺寸）
  async downloadImage(
    url: string,
    options: {
      width?: number;
      height?: number;
      quality?: number;
      format?: 'jpeg' | 'png' | 'webp';
    } = {}
  ): Promise<string> {
    const cacheKey = `${url}_${JSON.stringify(options)}`;

    // 检查缓存
    const cached = this.downloadCache.get(cacheKey);
    if (cached && !this.isExpired(cached)) {
      return cached.localPath;
    }

    // 检查是否已在下载中
    if (this.downloadQueue.has(cacheKey)) {
      return this.downloadQueue.get(cacheKey);
    }

    // 开始下载
    const downloadPromise = this.performDownload(url, cacheKey, options);
    this.downloadQueue.set(cacheKey, downloadPromise);

    try {
      return await downloadPromise;
    } finally {
      this.downloadQueue.delete(cacheKey);
    }
  }

  // 执行图片下载
  private async performDownload(
    url: string,
    cacheKey: string,
    options: any
  ): Promise<string> {
    try {
      // 构建优化后的URL
      const optimizedUrl = this.buildOptimizedUrl(url, options);

      // 下载文件
      const response = await fetch(optimizedUrl);
      if (!response.ok) {
        throw new Error(`Download failed: ${response.status}`);
      }

      // 获取文件数据
      const blob = await response.blob();

      // 保存到本地
      const localPath = await this.saveToCache(cacheKey, blob);

      // 更新缓存
      this.downloadCache.set(cacheKey, {
        localPath,
        timestamp: Date.now(),
        size: blob.size,
      });

      return localPath;
    } catch (error) {
      console.error('Image download failed:', error);
      throw error;
    }
  }

  // 构建优化URL
  private buildOptimizedUrl(url: string, options: any): string {
    const params = new URLSearchParams();

    if (options.width) params.append('w', options.width.toString());
    if (options.height) params.append('h', options.height.toString());
    if (options.quality) params.append('q', options.quality.toString());
    if (options.format) params.append('f', options.format);

    const paramString = params.toString();
    return paramString ? `${url}?${paramString}` : url;
  }

  // 保存到缓存
  private async saveToCache(key: string, blob: Blob): Promise<string> {
    // 这里需要使用文件系统API保存图片
    // 实际实现需要根据平台使用相应的文件系统库
    const fileName = `${key.replace(/[^a-zA-Z0-9]/g, '_')}.jpg`;

    // 模拟保存路径
    const localPath = `${FileSystem.cacheDirectory}images/${fileName}`;

    // 确保目录存在
    await FileSystem.makeDirectoryAsync(`${FileSystem.cacheDirectory}images`, {
      intermediates: true,
    });

    // 写入文件
    const base64 = await this.blobToBase64(blob);
    await FileSystem.writeAsStringAsync(localPath, base64, {
      encoding: FileSystem.EncodingType.Base64,
    });

    return localPath;
  }

  // Blob转Base64
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // 检查是否过期
  private isExpired(cached: CachedImage): boolean {
    const maxAge = 7 * 24 * 60 * 60 * 1000; // 7天
    return Date.now() - cached.timestamp > maxAge;
  }

  // 清理缓存
  cleanup() {
    const currentTime = Date.now();
    const maxAge = 7 * 24 * 60 * 60 * 1000; // 7天

    this.downloadCache.forEach((value, key) => {
      if (currentTime - value.timestamp > maxAge) {
        // 删除文件
        FileSystem.deleteAsync(value.localPath).catch(console.error);
        this.downloadCache.delete(key);
      }
    });
  }
}
```

---

## 🔋 电池优化

### 1. 电池监控

```typescript
// 电池监控管理器
class BatteryMonitor {
  private static instance: BatteryMonitor;
  private batteryLevel: number = 1.0;
  private isLowPowerMode: boolean = false;
  private listeners: Array<(info: BatteryInfo) => void> = [];

  static getInstance(): BatteryMonitor {
    if (!BatteryMonitor.instance) {
      BatteryMonitor.instance = new BatteryMonitor();
    }
    return BatteryMonitor.instance;
  }

  // 初始化电池监控
  async initialize() {
    if (Platform.OS === 'ios') {
      import('react-native').then(({ NativeModules }) => {
        if (NativeModules.BatteryMonitor) {
          NativeModules.BatteryMonitor.getBatteryLevel((level: number) => {
            this.batteryLevel = level;
            this.notifyListeners();
          });

          NativeModules.BatteryMonitor.getLowPowerMode((enabled: boolean) => {
            this.isLowPowerMode = enabled;
            this.notifyListeners();
          });

          // 监听电池状态变化
          NativeModules.BatteryMonitor.addBatteryLevelListener((level: number) => {
            this.batteryLevel = level;
            this.notifyListeners();
          });

          NativeModules.BatteryMonitor.addLowPowerModeListener((enabled: boolean) => {
            this.isLowPowerMode = enabled;
            this.notifyListeners();
          });
        }
      });
    }
  }

  // 添加监听器
  addListener(listener: (info: BatteryInfo) => void) {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  // 通知监听器
  private notifyListeners() {
    const batteryInfo: BatteryInfo = {
      level: this.batteryLevel,
      isLowPowerMode: this.isLowPowerMode,
      isLowBattery: this.batteryLevel < 0.2,
    };

    this.listeners.forEach(listener => {
      try {
        listener(batteryInfo);
      } catch (error) {
        console.error('Battery listener error:', error);
      }
    });
  }

  // 获取电池信息
  getBatteryInfo(): BatteryInfo {
    return {
      level: this.batteryLevel,
      isLowPowerMode: this.isLowPowerMode,
      isLowBattery: this.batteryLevel < 0.2,
    };
  }
}

// 电池感知的性能调节器
const useBatteryAwarePerformance = () => {
  const [batteryInfo, setBatteryInfo] = useState<BatteryInfo>({
    level: 1.0,
    isLowPowerMode: false,
    isLowBattery: false,
  });

  useEffect(() => {
    const batteryMonitor = BatteryMonitor.getInstance();

    batteryMonitor.initialize();
    const unsubscribe = batteryMonitor.addListener(setBatteryInfo);

    return unsubscribe;
  }, []);

  // 获取性能配置
  const getPerformanceConfig = useCallback(() => {
    if (batteryInfo.isLowPowerMode || batteryInfo.isLowBattery) {
      return {
        animationDuration: 200, // 减少动画时长
        enableHighQualityImages: false, // 降低图片质量
        maxConcurrentRequests: 2, // 减少并发请求
        enableBackgroundSync: false, // 禁用后台同步
        updateInterval: 30000, // 增加更新间隔
      };
    }

    return {
      animationDuration: 300,
      enableHighQualityImages: true,
      maxConcurrentRequests: 5,
      enableBackgroundSync: true,
      updateInterval: 10000,
    };
  }, [batteryInfo]);

  return {
    batteryInfo,
    performanceConfig: getPerformanceConfig(),
  };
};

// 电池优化的图片组件
const BatteryOptimizedImage = ({
  source,
  style,
  fallbackSource,
}: {
  source: ImageSourcePropType;
  style?: ImageStyle;
  fallbackSource?: ImageSourcePropType;
}) => {
  const { performanceConfig } = useBatteryAwarePerformance();
  const [optimizedSource, setOptimizedSource] = useState(source);

  useEffect(() => {
    if (!performanceConfig.enableHighQualityImages && fallbackSource) {
      setOptimizedSource(fallbackSource);
    } else {
      setOptimizedSource(source);
    }
  }, [performanceConfig, source, fallbackSource]);

  return (
    <Image
      source={optimizedSource}
      style={style}
      resizeMode="cover"
    />
  );
};
```

### 2. 后台任务优化

```typescript
// 后台任务管理器
class BackgroundTaskManager {
  private static instance: BackgroundTaskManager;
  private tasks: Map<string, BackgroundTask> = new Map();
  private isBackground = false;

  static getInstance(): BackgroundTaskManager {
    if (!BackgroundTaskManager.instance) {
      BackgroundTaskManager.instance = new BackgroundTaskManager();
    }
    return BackgroundTaskManager.instance;
  }

  // 初始化后台任务监听
  initialize() {
    AppState.addEventListener('change', this.handleAppStateChange);
  }

  // 处理应用状态变化
  handleAppStateChange = (nextAppState: string) => {
    if (nextAppState === 'background') {
      this.isBackground = true;
      this.suspendNonCriticalTasks();
    } else if (nextAppState === 'active') {
      this.isBackground = false;
      this.resumeTasks();
    }
  };

  // 暂停非关键任务
  private suspendNonCriticalTasks() {
    this.tasks.forEach((task, key) => {
      if (task.priority !== 'critical') {
        task.suspend();
      }
    });
  }

  // 恢复任务
  private resumeTasks() {
    this.tasks.forEach((task) => {
      task.resume();
    });
  }

  // 添加后台任务
  addTask(id: string, task: BackgroundTask) {
    this.tasks.set(id, task);

    // 如果当前在后台且任务不是关键的，则暂停
    if (this.isBackground && task.priority !== 'critical') {
      task.suspend();
    }
  }

  // 移除任务
  removeTask(id: string) {
    const task = this.tasks.get(id);
    if (task) {
      task.cancel();
      this.tasks.delete(id);
    }
  }

  // 获取任务状态
  getTaskStatus(id: string): TaskStatus | null {
    const task = this.tasks.get(id);
    return task ? task.getStatus() : null;
  }
}

// 智能数据同步器
class SmartDataSyncer {
  private static instance: SmartDataSyncer;
  private syncQueue: SyncTask[] = [];
  private isSyncing = false;
  private lastSyncTime = 0;

  static getInstance(): SmartDataSyncer {
    if (!SmartDataSyncer.instance) {
      SmartDataSyncer.instance = new SmartDataSyncer();
    }
    return SmartDataSyncer.instance;
  }

  // 添加同步任务
  addSyncTask(task: SyncTask) {
    this.syncQueue.push(task);
    this.scheduleSync();
  }

  // 调度同步
  private scheduleSync() {
    if (this.isSyncing) {
      return;
    }

    // 基于网络状态和电池状态决定是否立即同步
    const networkManager = NetworkPerformanceManager.getInstance();
    const batteryMonitor = BatteryMonitor.getInstance();

    const networkInfo = networkManager.getPerformanceReport();
    const batteryInfo = batteryMonitor.getBatteryInfo();

    // 如果网络条件不好或电量低，延迟同步
    const shouldDelay = networkInfo.errorRate > 0.1 ||
                        batteryInfo.isLowPowerMode ||
                        batteryInfo.isLowBattery;

    if (shouldDelay) {
      setTimeout(() => this.scheduleSync(), 60000); // 1分钟后重试
      return;
    }

    // 限制同步频率
    const now = Date.now();
    if (now - this.lastSyncTime < 30000) { // 30秒内最多同步一次
      return;
    }

    this.performSync();
  }

  // 执行同步
  private async performSync() {
    if (this.syncQueue.length === 0) {
      return;
    }

    this.isSyncing = true;
    this.lastSyncTime = Date.now();

    try {
      // 批量处理同步任务
      const batch = this.syncQueue.splice(0, 5); // 每次最多处理5个任务

      await Promise.allSettled(
        batch.map(task => task.execute())
      );

    } catch (error) {
      console.error('Sync failed:', error);
    } finally {
      this.isSyncing = false;

      // 如果还有任务，继续调度
      if (this.syncQueue.length > 0) {
        setTimeout(() => this.scheduleSync(), 5000);
      }
    }
  }

  // 强制同步
  async forceSync(): Promise<void> {
    this.lastSyncTime = 0; // 重置时间限制
    this.scheduleSync();
  }
}
```

---

## 📊 性能监控与分析

### 1. 性能监控Dashboard

```typescript
// 性能监控Hook
const usePerformanceMonitor = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    startup: { time: 0, status: 'pending' },
    render: { fps: 60, jank: 0, status: 'good' },
    memory: { used: 0, total: 0, percentage: 0, status: 'good' },
    network: { latency: 0, errorRate: 0, status: 'good' },
    battery: { level: 1.0, isLowPowerMode: false, status: 'good' },
  });

  // 更新启动性能
  const updateStartupMetrics = useCallback((time: number) => {
    setMetrics(prev => ({
      ...prev,
      startup: {
        time,
        status: time < 3000 ? 'good' : time < 5000 ? 'warning' : 'poor',
      },
    }));
  }, []);

  // 更新渲染性能
  const updateRenderMetrics = useCallback((fps: number, jank: number) => {
    setMetrics(prev => ({
      ...prev,
      render: {
        fps,
        jank,
        status: fps >= 55 && jank <= 5 ? 'good' : fps >= 30 && jank <= 15 ? 'warning' : 'poor',
      },
    }));
  }, []);

  // 更新内存使用
  const updateMemoryMetrics = useCallback((used: number, total: number) => {
    const percentage = (used / total) * 100;
    setMetrics(prev => ({
      ...prev,
      memory: {
        used,
        total,
        percentage,
        status: percentage < 70 ? 'good' : percentage < 90 ? 'warning' : 'poor',
      },
    }));
  }, []);

  // 更新网络性能
  const updateNetworkMetrics = useCallback((latency: number, errorRate: number) => {
    setMetrics(prev => ({
      ...prev,
      network: {
        latency,
        errorRate,
        status: latency < 500 && errorRate < 0.05 ? 'good' : latency < 1000 && errorRate < 0.1 ? 'warning' : 'poor',
      },
    }));
  }, []);

  return {
    metrics,
    updateStartupMetrics,
    updateRenderMetrics,
    updateMemoryMetrics,
    updateNetworkMetrics,
  };
};

// 性能报告组件
const PerformanceDashboard = ({ visible, onClose }: {
  visible: boolean;
  onClose: () => void;
}) => {
  const { metrics } = usePerformanceMonitor();

  if (!visible) return null;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return '#10B981';
      case 'warning': return '#F59E0B';
      case 'poor': return '#EF4444';
      default: return '#6B7280';
    }
  };

  return (
    <Modal visible={visible} transparent animationType="fade">
      <View style={styles.modalOverlay}>
        <View style={styles.dashboardContainer}>
          <View style={styles.dashboardHeader}>
            <Text style={styles.dashboardTitle}>性能监控</Text>
            <TouchableOpacity onPress={onClose}>
              <Icon name="close" size={24} color="#111827" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.metricsContainer}>
            {/* 启动性能 */}
            <MetricCard
              title="启动时间"
              value={`${metrics.startup.time}ms`}
              status={metrics.startup.status}
              color={getStatusColor(metrics.startup.status)}
            />

            {/* 渲染性能 */}
            <MetricCard
              title="渲染性能"
              value={`${metrics.render.fps} FPS`}
              subtitle={`掉帧: ${metrics.render.jank}`}
              status={metrics.render.status}
              color={getStatusColor(metrics.render.status)}
            />

            {/* 内存使用 */}
            <MetricCard
              title="内存使用"
              value={`${Math.round(metrics.memory.percentage)}%`}
              subtitle={`${Math.round(metrics.memory.used / 1024 / 1024)}MB`}
              status={metrics.memory.status}
              color={getStatusColor(metrics.memory.status)}
            />

            {/* 网络性能 */}
            <MetricCard
              title="网络延迟"
              value={`${metrics.network.latency}ms`}
              subtitle={`错误率: ${(metrics.network.errorRate * 100).toFixed(1)}%`}
              status={metrics.network.status}
              color={getStatusColor(metrics.network.status)}
            />

            {/* 电池状态 */}
            <MetricCard
              title="电池电量"
              value={`${Math.round(metrics.battery.level * 100)}%`}
              subtitle={metrics.battery.isLowPowerMode ? '低功耗模式' : '正常模式'}
              status={metrics.battery.status}
              color={getStatusColor(metrics.battery.status)}
            />
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

// 指标卡片组件
const MetricCard = ({
  title,
  value,
  subtitle,
  status,
  color,
}: {
  title: string;
  value: string;
  subtitle?: string;
  status: string;
  color: string;
}) => (
  <View style={styles.metricCard}>
    <View style={styles.metricHeader}>
      <Text style={styles.metricTitle}>{title}</Text>
      <View style={[styles.statusIndicator, { backgroundColor: color }]} />
    </View>
    <Text style={styles.metricValue}>{value}</Text>
    {subtitle && <Text style={styles.metricSubtitle}>{subtitle}</Text>}
  </View>
);
```

### 2. 性能分析工具

```typescript
// 性能分析器
class PerformanceProfiler {
  private static instance: PerformanceProfiler;
  private profiles: Map<string, Profile> = new Map();

  static getInstance(): PerformanceProfiler {
    if (!PerformanceProfiler.instance) {
      PerformanceProfiler.instance = new PerformanceProfiler();
    }
    return PerformanceProfiler.instance;
  }

  // 开始性能分析
  startProfile(name: string) {
    const profile: Profile = {
      name,
      startTime: performance.now(),
      endTime: 0,
      duration: 0,
      samples: [],
      memorySnapshots: [],
    };

    this.profiles.set(name, profile);
  }

  // 添加样本
  addSample(name: string, data: any) {
    const profile = this.profiles.get(name);
    if (profile) {
      profile.samples.push({
        timestamp: performance.now(),
        data,
      });
    }
  }

  // 添加内存快照
  addMemorySnapshot(name: string) {
    const profile = this.profiles.get(name);
    if (profile) {
      const memoryMonitor = MemoryMonitor.getInstance();
      const currentMemory = memoryMonitor.getCurrentMemoryUsage();

      if (currentMemory) {
        profile.memorySnapshots.push({
          timestamp: performance.now(),
          memory: currentMemory,
        });
      }
    }
  }

  // 结束性能分析
  endProfile(name: string): Profile | null {
    const profile = this.profiles.get(name);
    if (profile) {
      profile.endTime = performance.now();
      profile.duration = profile.endTime - profile.startTime;

      // 分析性能数据
      profile.analysis = this.analyzeProfile(profile);

      return profile;
    }
    return null;
  }

  // 分析性能数据
  private analyzeProfile(profile: Profile): ProfileAnalysis {
    const duration = profile.duration;
    const sampleCount = profile.samples.length;
    const memorySnapshots = profile.memorySnapshots;

    // 计算平均采样间隔
    let avgSampleInterval = 0;
    if (sampleCount > 1) {
      const intervals = [];
      for (let i = 1; i < sampleCount; i++) {
        intervals.push(profile.samples[i].timestamp - profile.samples[i - 1].timestamp);
      }
      avgSampleInterval = intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length;
    }

    // 分析内存使用趋势
    let memoryTrend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (memorySnapshots.length > 1) {
      const first = memorySnapshots[0].memory.percentage;
      const last = memorySnapshots[memorySnapshots.length - 1].memory.percentage;

      if (last > first + 5) {
        memoryTrend = 'increasing';
      } else if (last < first - 5) {
        memoryTrend = 'decreasing';
      }
    }

    // 找出内存峰值
    const maxMemory = memorySnapshots.reduce((max, snapshot) => {
      return Math.max(max, snapshot.memory.percentage);
    }, 0);

    return {
      avgSampleInterval,
      memoryTrend,
      maxMemoryUsage: maxMemory,
      recommendations: this.generateRecommendations(profile),
    };
  }

  // 生成优化建议
  private generateRecommendations(profile: Profile): string[] {
    const recommendations: string[] = [];

    if (profile.duration > 1000) {
      recommendations.push('操作耗时过长，考虑异步处理或优化算法');
    }

    const analysis = profile.analysis;
    if (analysis.memoryTrend === 'increasing') {
      recommendations.push('内存使用持续增长，检查是否存在内存泄漏');
    }

    if (analysis.maxMemoryUsage > 80) {
      recommendations.push('内存使用峰值过高，考虑优化数据结构或增加清理机制');
    }

    if (analysis.avgSampleInterval > 100) {
      recommendations.push('采样间隔较大，可能存在性能瓶颈');
    }

    return recommendations;
  }

  // 获取性能报告
  getReport(name: string): PerformanceReport | null {
    const profile = this.profiles.get(name);
    if (!profile) return null;

    return {
      name: profile.name,
      duration: profile.duration,
      sampleCount: profile.samples.length,
      memorySnapshots: profile.memorySnapshots.length,
      analysis: profile.analysis,
    };
  }
}
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
- [07-响应式设计](07-responsive-design.md)：响应式设计策略

### 开发文档系列
- [09-测试策略](09-testing-strategy.md)：测试策略和工具
- [10-部署指南](10-deployment-guide.md)：应用打包和发布

### 技术参考
- [React Native性能优化](https://reactnative.dev/docs/performance)
- [React Native性能监控](https://reactnative.dev/docs/debugging-performance)
- [Android性能最佳实践](https://developer.android.com/topic/performance)
- [iOS性能指南](https://developer.apple.com/documentation/xcode/improving_your_app_s_performance)

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**下次更新**: 根据性能监控数据持续优化改进