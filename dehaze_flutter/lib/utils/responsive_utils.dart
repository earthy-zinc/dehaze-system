import 'package:flutter/material.dart';

/// 响应式断点定义
/// 与 Tailwind CSS 断点保持一致，符合 Material Design 推荐
class Breakpoints {
  static const double xs = 0; // 手机竖屏
  static const double sm = 640; // 手机横屏
  static const double md = 768; // 平板
  static const double lg = 1024; // 小桌面
  static const double xl = 1280; // 桌面
  static const double xxl = 1536; // 大桌面

  /// Material Design 3 推荐断点
  static const double compact = 600; // 紧凑型（手机）
  static const double medium = 840; // 中等（平板）
  static const double expanded = 1200; // 扩展型（桌面）
}

/// 设备类型枚举
enum DeviceType { mobile, tablet, desktop }

/// 响应式工具类
///
/// 遵循 Flutter 官方响应式最佳实践：
/// - 使用 MediaQuery.sizeOf() 而非 MediaQuery.of() 避免不必要的重建
/// - 基于窗口尺寸而非设备类型判断布局
/// - 不锁定屏幕方向
class ResponsiveUtils {
  /// 获取当前设备类型（基于窗口宽度，非硬件类型）
  static DeviceType getDeviceType(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    if (width < Breakpoints.md) return DeviceType.mobile;
    if (width < Breakpoints.lg) return DeviceType.tablet;
    return DeviceType.desktop;
  }

  /// 判断是否为移动设备尺寸
  static bool isMobile(BuildContext context) =>
      MediaQuery.sizeOf(context).width < Breakpoints.md;

  /// 判断是否为平板设备尺寸
  static bool isTablet(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    return width >= Breakpoints.md && width < Breakpoints.lg;
  }

  /// 判断是否为桌面设备尺寸
  static bool isDesktop(BuildContext context) =>
      MediaQuery.sizeOf(context).width >= Breakpoints.lg;

  /// 判断是否为宽屏（平板或桌面）
  static bool isWideScreen(BuildContext context) =>
      MediaQuery.sizeOf(context).width >= Breakpoints.md;

  /// 获取当前窗口宽度
  static double getWindowWidth(BuildContext context) =>
      MediaQuery.sizeOf(context).width;

  /// 获取当前窗口高度
  static double getWindowHeight(BuildContext context) =>
      MediaQuery.sizeOf(context).height;

  /// 获取响应式网格列数
  /// 用于图片网格、卡片网格等
  static int getGridCrossAxisCount(
    BuildContext context, {
    int mobile = 2,
    int tablet = 3,
    int desktop = 4,
    int largeDesktop = 5,
  }) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= Breakpoints.xl) return largeDesktop;
    if (width >= Breakpoints.lg) return desktop;
    if (width >= Breakpoints.md) return tablet;
    return mobile;
  }

  /// 获取瀑布流列数（与 CSS column-count 对应）
  static int getWaterfallColumnCount(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= Breakpoints.lg) return 4;
    if (width >= Breakpoints.sm) return 3;
    return 2;
  }

  /// 获取响应式间距
  static double getResponsiveSpacing(
    BuildContext context, {
    double mobile = 12,
    double tablet = 16,
    double desktop = 20,
  }) {
    final deviceType = getDeviceType(context);
    switch (deviceType) {
      case DeviceType.mobile:
        return mobile;
      case DeviceType.tablet:
        return tablet;
      case DeviceType.desktop:
        return desktop;
    }
  }

  /// 获取响应式字体大小
  static double getResponsiveFontSize(
    BuildContext context, {
    required double base,
    double mobileScale = 0.85,
    double tabletScale = 0.95,
  }) {
    final deviceType = getDeviceType(context);
    switch (deviceType) {
      case DeviceType.mobile:
        return base * mobileScale;
      case DeviceType.tablet:
        return base * tabletScale;
      case DeviceType.desktop:
        return base;
    }
  }

  /// 获取响应式内边距
  static EdgeInsets getResponsivePadding(BuildContext context) {
    final deviceType = getDeviceType(context);
    switch (deviceType) {
      case DeviceType.mobile:
        return const EdgeInsets.all(12);
      case DeviceType.tablet:
        return const EdgeInsets.all(16);
      case DeviceType.desktop:
        return const EdgeInsets.all(24);
    }
  }

  /// 获取卡片宽高比
  static double getCardAspectRatio(
    BuildContext context, {
    double mobile = 1.0,
    double tablet = 1.1,
    double desktop = 1.2,
  }) {
    final deviceType = getDeviceType(context);
    switch (deviceType) {
      case DeviceType.mobile:
        return mobile;
      case DeviceType.tablet:
        return tablet;
      case DeviceType.desktop:
        return desktop;
    }
  }

  /// 根据宽度获取响应式值
  /// 适用于需要更精细控制的场景
  static T getResponsiveValue<T>(
    BuildContext context, {
    required T defaultValue,
    T? xs,
    T? sm,
    T? md,
    T? lg,
    T? xl,
    T? xxl,
  }) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= Breakpoints.xxl) return xxl ?? xl ?? lg ?? md ?? sm ?? xs ?? defaultValue;
    if (width >= Breakpoints.xl) return xl ?? lg ?? md ?? sm ?? xs ?? defaultValue;
    if (width >= Breakpoints.lg) return lg ?? md ?? sm ?? xs ?? defaultValue;
    if (width >= Breakpoints.md) return md ?? sm ?? xs ?? defaultValue;
    if (width >= Breakpoints.sm) return sm ?? xs ?? defaultValue;
    return xs ?? defaultValue;
  }
}

/// 响应式构建器 Widget
///
/// 根据窗口宽度自动选择合适的布局
class ResponsiveBuilder extends StatelessWidget {
  const ResponsiveBuilder({
    required this.mobile,
    super.key,
    this.tablet,
    this.desktop,
  });

  final Widget mobile;
  final Widget? tablet;
  final Widget? desktop;

  @override
  Widget build(BuildContext context) {
    final deviceType = ResponsiveUtils.getDeviceType(context);

    switch (deviceType) {
      case DeviceType.desktop:
        return desktop ?? tablet ?? mobile;
      case DeviceType.tablet:
        return tablet ?? mobile;
      case DeviceType.mobile:
        return mobile;
    }
  }
}

/// 响应式布局约束
///
/// 限制内容最大宽度，避免在大屏幕上内容过度拉伸
class ResponsiveConstraints extends StatelessWidget {
  const ResponsiveConstraints({
    required this.child,
    super.key,
    this.maxWidth = 1200,
    this.padding,
    this.alignment = Alignment.topCenter,
  });

  final Widget child;
  final double maxWidth;
  final EdgeInsets? padding;
  final Alignment alignment;

  @override
  Widget build(BuildContext context) => Align(
        alignment: alignment,
        child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: maxWidth),
          child: Padding(
            padding:
                padding ?? ResponsiveUtils.getResponsivePadding(context),
            child: child,
          ),
        ),
      );
}

/// 自适应网格视图
///
/// 根据可用宽度自动计算列数，避免固定列数导致的布局问题
class AdaptiveGridView extends StatelessWidget {
  const AdaptiveGridView({
    required this.children,
    super.key,
    this.minItemWidth = 150,
    this.maxItemWidth = 300,
    this.spacing = 16,
    this.childAspectRatio = 1.0,
    this.padding,
    this.shrinkWrap = false,
    this.physics,
  });

  final List<Widget> children;
  final double minItemWidth;
  final double maxItemWidth;
  final double spacing;
  final double childAspectRatio;
  final EdgeInsets? padding;
  final bool shrinkWrap;
  final ScrollPhysics? physics;

  @override
  Widget build(BuildContext context) => LayoutBuilder(
        builder: (context, constraints) {
          // 根据可用宽度计算最佳列数
          final availableWidth = constraints.maxWidth - (padding?.horizontal ?? 0);
          var crossAxisCount = (availableWidth / minItemWidth).floor();
          crossAxisCount = crossAxisCount.clamp(1, (availableWidth / maxItemWidth).ceil().clamp(1, 10));

          return GridView.builder(
            padding: padding,
            shrinkWrap: shrinkWrap,
            physics: physics,
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: crossAxisCount,
              crossAxisSpacing: spacing,
              mainAxisSpacing: spacing,
              childAspectRatio: childAspectRatio,
            ),
            itemCount: children.length,
            itemBuilder: (context, index) => children[index],
          );
        },
      );
}
