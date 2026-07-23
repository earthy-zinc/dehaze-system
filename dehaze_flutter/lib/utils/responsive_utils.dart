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

/// 响应式工具类
///
/// 遵循 Flutter 官方响应式最佳实践：
/// - 使用 MediaQuery.sizeOf() 而非 MediaQuery.of() 避免不必要的重建
/// - 基于窗口尺寸而非设备类型判断布局
/// - 不锁定屏幕方向
class ResponsiveUtils {
  /// 判断是否为移动设备尺寸
  static bool isMobile(BuildContext context) =>
      MediaQuery.sizeOf(context).width < Breakpoints.md;

  /// 判断是否为宽屏（平板或桌面）
  static bool isWideScreen(BuildContext context) =>
      MediaQuery.sizeOf(context).width >= Breakpoints.md;

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

  /// 获取响应式间距
  static double getResponsiveSpacing(
    BuildContext context, {
    double mobile = 12,
    double tablet = 16,
    double desktop = 20,
  }) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= Breakpoints.lg) return desktop;
    if (width >= Breakpoints.md) return tablet;
    return mobile;
  }

  /// 获取响应式内边距
  static EdgeInsets getResponsivePadding(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= Breakpoints.lg) return const EdgeInsets.all(24);
    if (width >= Breakpoints.md) return const EdgeInsets.all(16);
    return const EdgeInsets.all(12);
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
