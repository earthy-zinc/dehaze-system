import 'package:flutter/material.dart';

/// Dehaze Flutter 应用主题配置
///
/// 基于设计系统文档的最佳实践主题配置，支持明亮和暗黑两种模式
/// 与 HTML 设计稿 (demo/index.html) 保持一致的视觉规范
class AppTheme {
  // ==================== 颜色系统定义 ====================

  // 品牌色彩 - 与 Tailwind CSS 设计稿保持一致
  static const Color brandBlue = Color(0xFF3b82f6); // Tailwind blue-500
  static const Color brandBlueDark = Color(0xFF2563eb); // Tailwind blue-600
  static const Color brandBlueLight = Color(0xFF60a5fa); // Tailwind blue-400
  static const Color techGreen = Color(0xFF10b981); // Tailwind emerald-500
  static const Color accentGreen = Color(0xFF34d399); // Tailwind emerald-400

  // 功能色彩
  static const Color successColor = Color(0xFF34A853); // 成功
  static const Color warningColor = Color(0xFFFF9800); // 警告
  static const Color errorColor = Color(0xFFEA4335); // 错误
  static const Color infoColor = Color(0xFF4285F4); // 信息

  // 中性色彩 - 明亮主题
  static const Color lightTextPrimary = Color(0xFF212121); // 主要文字
  static const Color lightTextSecondary = Color(0xFF757575); // 次要文字
  static const Color lightTextDisabled = Color(0xFFBDBDBD); // 禁用文字

  static const Color lightBgPrimary = Color(0xFFFFFFFF); // 主背景
  static const Color lightBgSecondary = Color(0xFFF8F9FA); // 次背景

  static const Color lightBorderPrimary = Color(0xFFE0E0E0); // 主边框
  static const Color lightBorderSecondary = Color(0xFFF5F5F5); // 次边框

  // 中性色彩 - 暗黑主题
  static const Color darkTextPrimary = Color(0xFFE1E2E1); // 主要文字
  static const Color darkTextSecondary = Color(0xFF9E9E9E); // 次要文字
  static const Color darkTextDisabled = Color(0xFF616161); // 禁用文字

  static const Color darkBgPrimary = Color(0xFF121212); // 主背景
  static const Color darkBgSecondary = Color(0xFF1E1E1E); // 次背景

  static const Color darkBorderPrimary = Color(0xFF424242); // 主边框
  static const Color darkBorderSecondary = Color(0xFF2A2A2A); // 次边框

  // 渐变色
  static const List<Color> primaryGradient = [
    Color(0xFF4285F4),
    Color(0xFF6366F1),
  ];

  static const List<Color> secondaryGradient = [
    Color(0xFF667EEA),
    Color(0xFF764BA2),
  ];

  static const List<Color> functionalGradient = [
    Color(0xFFF093FB),
    Color(0xFFF5576C),
  ];

  // Hero 区域渐变色 - 与设计稿一致
  static const List<Color> heroGradient = [
    Color(0xFF1e40af), // blue-800
    Color(0xFF3b82f6), // blue-500
    Color(0xFF60a5fa), // blue-400
  ];

  // 工具卡片背景渐变
  static const List<Color> toolCardGradient = [
    Color(0xFFeff6ff), // blue-50
    Color(0xFFdbeafe), // blue-100
  ];

  // ==================== 间距系统定义 ====================
  static double get spacingXS => 4; // 超小间距
  static double get spacingS => 8; // 小间距
  static double get spacingM => 16; // 标准间距
  static double get spacingL => 24; // 大间距
  static double get spacingXL => 32; // 超大间距
  static double get spacingXXL => 48; // 超超大间距

  // ==================== 圆角系统定义 ====================
  static double get radiusXS => 2; // 超小圆角
  static double get radiusS => 4; // 小圆角
  static double get radiusM => 8; // 标准圆角
  static double get radiusL => 16; // 大圆角
  static double get radiusXL => 24; // 超大圆角

  // ==================== 阴影系统定义 ====================
  static const List<BoxShadow> shadowLevel1 = [
    BoxShadow(
      offset: Offset(0, 1),
      blurRadius: 3,
      color: Color(0x1F000000), // 12% opacity
    ),
  ];

  static const List<BoxShadow> shadowLevel2 = [
    BoxShadow(
      offset: Offset(0, 2),
      blurRadius: 8,
      color: Color(0x26000000), // 15% opacity
    ),
  ];

  static const List<BoxShadow> shadowLevel3 = [
    BoxShadow(
      offset: Offset(0, 4),
      blurRadius: 12,
      color: Color(0x2E000000), // 18% opacity
    ),
  ];

  static const List<BoxShadow> shadowLevel4 = [
    BoxShadow(
      offset: Offset(0, 8),
      blurRadius: 24,
      color: Color(0x38000000), // 22% opacity
    ),
  ];

  static const List<BoxShadow> shadowLevel5 = [
    BoxShadow(
      offset: Offset(0, 16),
      blurRadius: 48,
      color: Color(0x47000000), // 28% opacity
    ),
  ];

  // ==================== 明亮主题 ====================

  static ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    fontFamily: 'NotoSansSC',

    // 颜色方案
    colorScheme: const ColorScheme.light(
      primary: brandBlue,
      secondary: techGreen,
      tertiary: accentGreen,
      error: errorColor,
      surface: lightBgPrimary,
      onPrimary: Colors.white,
      onSecondary: Colors.white,
      onTertiary: Colors.white,
      onError: Colors.white,
      onSurface: lightTextPrimary,
      outline: lightBorderPrimary,
      surfaceContainerHighest: lightBgSecondary,
      onSurfaceVariant: lightTextSecondary,
    ),

    // 文字主题
    textTheme: _buildTextTheme(lightTextPrimary, lightTextSecondary),

    // 应用栏主题
    appBarTheme: AppBarTheme(
      backgroundColor: lightBgPrimary,
      foregroundColor: lightTextPrimary,
      elevation: 0,
      scrolledUnderElevation: 1,
      centerTitle: true,
      titleTextStyle: TextStyle(
        fontSize: 20, // H3 size
        fontWeight: FontWeight.w500,
        color: lightTextPrimary,
        letterSpacing: 0,
      ),
      iconTheme: IconThemeData(color: lightTextPrimary, size: 24),
    ),

    // 卡片主题
    cardTheme: CardThemeData(
      color: lightBgPrimary,
      elevation: 2,
      shadowColor: Colors.black.withValues(alpha: 0.08),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusL),
      ),
      margin: EdgeInsets.symmetric(horizontal: spacingM, vertical: spacingS),
    ),

    // 按钮主题
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: brandBlue,
        foregroundColor: Colors.white,
        elevation: 2,
        shadowColor: brandBlue.withValues(alpha: 0.3),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusM),
        ),
        padding: EdgeInsets.symmetric(horizontal: spacingL, vertical: spacingM),
        textStyle: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
      ),
    ),

    // 文本按钮主题
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: brandBlue,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusM),
        ),
        padding: EdgeInsets.symmetric(horizontal: spacingM, vertical: spacingS),
        textStyle: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
      ),
    ),

    // 轮廓按钮主题
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: brandBlue,
        side: const BorderSide(color: brandBlue, width: 1),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusM),
        ),
        padding: EdgeInsets.symmetric(horizontal: spacingL, vertical: spacingM),
        textStyle: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
      ),
    ),

    // 输入框主题
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: lightBgPrimary,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: lightBorderPrimary),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: lightBorderPrimary),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: brandBlue, width: 2),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: errorColor, width: 1),
      ),
      contentPadding: EdgeInsets.symmetric(
        horizontal: spacingM,
        vertical: spacingM,
      ),
      hintStyle: TextStyle(color: lightTextDisabled, fontSize: 16),
      labelStyle: TextStyle(color: lightTextSecondary, fontSize: 14),
    ),

    // 图标主题
    iconTheme: IconThemeData(color: lightTextPrimary, size: 24),

    // 分割线主题
    dividerTheme: const DividerThemeData(
      color: lightBorderPrimary,
      thickness: 1,
    ),

    // 底部导航栏主题
    bottomNavigationBarTheme: BottomNavigationBarThemeData(
      backgroundColor: lightBgPrimary,
      selectedItemColor: brandBlue,
      unselectedItemColor: lightTextSecondary,
      selectedLabelStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
      unselectedLabelStyle: TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w400,
      ),
      type: BottomNavigationBarType.fixed,
      elevation: 3,
    ),

    // 浮动操作按钮主题
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: brandBlue,
      foregroundColor: Colors.white,
      elevation: 3,
    ),

    // 开关主题
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return brandBlue;
        }
        return const Color(0xFFBDBDBD);
      }),
      trackColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return brandBlue.withValues(alpha: 0.5);
        }
        return const Color(0xFFE0E0E0);
      }),
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    ),

    // 复选框主题
    checkboxTheme: CheckboxThemeData(
      fillColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return brandBlue;
        }
        return Colors.transparent;
      }),
      checkColor: WidgetStateProperty.all(Colors.white),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusXS),
      ),
    ),

    // 单选框主题
    radioTheme: RadioThemeData(
      fillColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return brandBlue;
        }
        return const Color(0xFF757575);
      }),
    ),

    // 芯片标签主题
    chipTheme: ChipThemeData(
      backgroundColor: lightBgSecondary,
      selectedColor: brandBlue.withValues(alpha: 0.1),
      disabledColor: lightBorderPrimary,
      labelStyle: TextStyle(
        color: lightTextPrimary,
        fontSize: 12,
        fontWeight: FontWeight.w500,
      ),
      secondaryLabelStyle: TextStyle(
        color: brandBlue,
        fontSize: 12,
        fontWeight: FontWeight.w500,
      ),
      padding: EdgeInsets.symmetric(horizontal: spacingS, vertical: spacingXS),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusS),
      ),
    ),

    // 滑块主题
    sliderTheme: SliderThemeData(
      activeTrackColor: brandBlue,
      inactiveTrackColor: lightBorderPrimary,
      thumbColor: brandBlue,
      overlayColor: brandBlue.withValues(alpha: 0.2),
      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 10),
      overlayShape: const RoundSliderOverlayShape(overlayRadius: 20),
      valueIndicatorColor: brandBlue,
      valueIndicatorTextStyle: TextStyle(color: Colors.white, fontSize: 14),
    ),

    // 进度条主题
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: brandBlue,
      linearTrackColor: lightBorderPrimary,
      circularTrackColor: lightBorderPrimary,
    ),
  );

  // ==================== 暗黑主题 ====================

  static ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    fontFamily: 'NotoSansSC',

    // 颜色方案
    colorScheme: const ColorScheme.dark(
      primary: Color(0xFF90CAF9), // 更亮的主色
      secondary: Color(0xFF81C784), // 更亮的次要色
      tertiary: Color(0xFFA5D6A7), // 更亮的第三色
      error: Color(0xFFEF5350), // 更亮的错误色
      surface: darkBgPrimary,
      onPrimary: Color(0xFF1C1B1F), // 主色上的文字颜色
      onSecondary: Color(0xFF1C1B1F), // 次要色上的文字颜色
      onTertiary: Color(0xFF1C1B1F), // 第三色上的文字颜色
      onError: Colors.white, // 错误色上的文字颜色
      onSurface: darkTextPrimary,
      outline: darkBorderPrimary,
      surfaceContainerHighest: darkBgSecondary,
      onSurfaceVariant: darkTextSecondary,
    ),

    // 文字主题
    textTheme: _buildTextTheme(darkTextPrimary, darkTextSecondary),

    // 应用栏主题
    appBarTheme: AppBarTheme(
      backgroundColor: darkBgPrimary,
      foregroundColor: darkTextPrimary,
      elevation: 0,
      scrolledUnderElevation: 1,
      centerTitle: true,
      titleTextStyle: TextStyle(
        fontSize: 20, // H3 size
        fontWeight: FontWeight.w500,
        color: darkTextPrimary,
        letterSpacing: 0,
      ),
      iconTheme: IconThemeData(color: darkTextPrimary, size: 24),
    ),

    // 卡片主题
    cardTheme: CardThemeData(
      color: darkBgSecondary,
      elevation: 4,
      shadowColor: Colors.black.withValues(alpha: 0.3),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusL),
      ),
      margin: EdgeInsets.symmetric(horizontal: spacingM, vertical: spacingS),
    ),

    // 按钮主题
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: const Color(0xFF90CAF9),
        foregroundColor: const Color(0xFF1C1B1F),
        elevation: 3,
        shadowColor: const Color(0xFF90CAF9).withValues(alpha: 0.4),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusM),
        ),
        padding: EdgeInsets.symmetric(horizontal: spacingL, vertical: spacingM),
        textStyle: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
      ),
    ),

    // 文本按钮主题
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: const Color(0xFF90CAF9),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusM),
        ),
        padding: EdgeInsets.symmetric(horizontal: spacingM, vertical: spacingS),
        textStyle: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
      ),
    ),

    // 轮廓按钮主题
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: const Color(0xFF90CAF9),
        side: const BorderSide(color: Color(0xFF90CAF9), width: 1),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusM),
        ),
        padding: EdgeInsets.symmetric(horizontal: spacingL, vertical: spacingM),
        textStyle: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
      ),
    ),

    // 输入框主题
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: darkBgSecondary,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: darkBorderPrimary),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: darkBorderPrimary),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: Color(0xFF90CAF9), width: 2),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusM),
        borderSide: const BorderSide(color: Color(0xFFEF5350), width: 1),
      ),
      contentPadding: EdgeInsets.symmetric(
        horizontal: spacingM,
        vertical: spacingM,
      ),
      hintStyle: TextStyle(color: darkTextDisabled, fontSize: 16),
      labelStyle: TextStyle(color: darkTextSecondary, fontSize: 14),
    ),

    // 图标主题
    iconTheme: IconThemeData(color: darkTextPrimary, size: 24),

    // 分割线主题
    dividerTheme: const DividerThemeData(
      color: darkBorderPrimary,
      thickness: 1,
    ),

    // 底部导航栏主题
    bottomNavigationBarTheme: BottomNavigationBarThemeData(
      backgroundColor: darkBgPrimary,
      selectedItemColor: Color(0xFF90CAF9),
      unselectedItemColor: darkTextSecondary,
      selectedLabelStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
      unselectedLabelStyle: TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w400,
      ),
      type: BottomNavigationBarType.fixed,
      elevation: 3,
    ),

    // 浮动操作按钮主题
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: Color(0xFF90CAF9),
      foregroundColor: Color(0xFF1C1B1F),
      elevation: 6,
    ),

    // 开关主题
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return const Color(0xFF90CAF9);
        }
        return const Color(0xFF616161);
      }),
      trackColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return const Color(0xFF90CAF9).withValues(alpha: 0.5);
        }
        return const Color(0xFF424242);
      }),
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    ),

    // 复选框主题
    checkboxTheme: CheckboxThemeData(
      fillColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return const Color(0xFF90CAF9);
        }
        return Colors.transparent;
      }),
      checkColor: WidgetStateProperty.all(const Color(0xFF1C1B1F)),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusXS),
      ),
    ),

    // 单选框主题
    radioTheme: RadioThemeData(
      fillColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return const Color(0xFF90CAF9);
        }
        return const Color(0xFF9E9E9E);
      }),
    ),

    // 芯片标签主题
    chipTheme: ChipThemeData(
      backgroundColor: darkBgSecondary,
      selectedColor: const Color(0xFF90CAF9).withValues(alpha: 0.2),
      disabledColor: darkBorderPrimary,
      labelStyle: TextStyle(
        color: darkTextPrimary,
        fontSize: 12,
        fontWeight: FontWeight.w500,
      ),
      secondaryLabelStyle: TextStyle(
        color: Color(0xFF90CAF9),
        fontSize: 12,
        fontWeight: FontWeight.w500,
      ),
      padding: EdgeInsets.symmetric(horizontal: spacingS, vertical: spacingXS),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusS),
      ),
    ),

    // 滑块主题
    sliderTheme: SliderThemeData(
      activeTrackColor: const Color(0xFF90CAF9),
      inactiveTrackColor: darkBorderPrimary,
      thumbColor: const Color(0xFF90CAF9),
      overlayColor: const Color(0xFF90CAF9).withValues(alpha: 0.2),
      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 10),
      overlayShape: const RoundSliderOverlayShape(overlayRadius: 20),
      valueIndicatorColor: const Color(0xFF90CAF9),
      valueIndicatorTextStyle: TextStyle(
        color: const Color(0xFF1C1B1F),
        fontSize: 14,
      ),
    ),

    // 进度条主题
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: Color(0xFF90CAF9),
      linearTrackColor: darkBorderPrimary,
      circularTrackColor: darkBorderPrimary,
    ),
  );

  // ==================== 文字主题构建 ====================

  static TextTheme _buildTextTheme(Color primaryColor, Color secondaryColor) =>
      TextTheme(
        // 显示文字 (Display) - 响应式字体大小
        displayLarge: TextStyle(
          inherit: true,
          fontSize: 32, // H1
          fontWeight: FontWeight.w700,
          letterSpacing: -0.25,
          height: 1.2,
          color: primaryColor,
        ),
        displayMedium: TextStyle(
          inherit: true,
          fontSize: 24, // H2
          fontWeight: FontWeight.w500,
          height: 1.3,
          color: primaryColor,
        ),
        displaySmall: TextStyle(
          inherit: true,
          fontSize: 20, // H3
          fontWeight: FontWeight.w500,
          height: 1.3,
          color: primaryColor,
        ),

        // 标题文字 (Headline) - 响应式字体大小
        headlineLarge: TextStyle(
          inherit: true,
          fontSize: 48, // H1
          fontWeight: FontWeight.w900,
          height: 1.5,
          color: primaryColor,
        ),
        headlineMedium: TextStyle(
          inherit: true,
          fontSize: 40, // H2
          fontWeight: FontWeight.w900,
          height: 1.3,
          color: primaryColor,
        ),
        headlineSmall: TextStyle(
          inherit: true,
          fontSize: 32, // H3
          fontWeight: FontWeight.w900,
          height: 1.3,
          color: primaryColor,
        ),

        // 标题文字 (Title) - 响应式字体大小
        titleLarge: TextStyle(
          inherit: true,
          fontSize: 18, // H4
          fontWeight: FontWeight.w400,
          height: 1.4,
          color: primaryColor,
        ),
        titleMedium: TextStyle(
          inherit: true,
          fontSize: 16, // Body Large
          fontWeight: FontWeight.w400,
          height: 1.5,
          color: primaryColor,
        ),
        titleSmall: TextStyle(
          inherit: true,
          fontSize: 14, // Body
          fontWeight: FontWeight.w400,
          height: 1.5,
          color: primaryColor,
        ),

        // 正文文字 (Body) - 响应式字体大小
        bodyLarge: TextStyle(
          inherit: true,
          fontSize: 16, // Body Large
          fontWeight: FontWeight.w400,
          height: 1.5,
          color: primaryColor,
        ),
        bodyMedium: TextStyle(
          inherit: true,
          fontSize: 14, // Body
          fontWeight: FontWeight.w400,
          height: 1.5,
          color: primaryColor,
        ),
        bodySmall: TextStyle(
          inherit: true,
          fontSize: 12, // Body Small
          fontWeight: FontWeight.w400,
          height: 1.4,
          color: secondaryColor,
        ),

        // 标签文字 (Label) - 响应式字体大小
        labelLarge: TextStyle(
          inherit: true,
          fontSize: 14, // Body
          fontWeight: FontWeight.w500,
          height: 1.5,
          color: primaryColor,
        ),
        labelMedium: TextStyle(
          inherit: true,
          fontSize: 12, // Body Small
          fontWeight: FontWeight.w500,
          height: 1.4,
          color: secondaryColor,
        ),
        labelSmall: TextStyle(
          inherit: true,
          fontSize: 10, // Caption
          fontWeight: FontWeight.w400,
          height: 1.4,
          color: secondaryColor,
        ),
      );

  // ==================== 主题扩展 ====================

  /// 获取渐变背景
  static LinearGradient getPrimaryGradient() => const LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: primaryGradient,
  );

  static LinearGradient getSecondaryGradient() => const LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: secondaryGradient,
  );

  /// 根据主题模式获取适当的阴影
  static List<BoxShadow> getShadow(int level, {bool isDark = false}) {
    switch (level) {
      case 1:
        return shadowLevel1;
      case 2:
        return shadowLevel2;
      case 3:
        return shadowLevel3;
      case 4:
        return shadowLevel4;
      case 5:
        return shadowLevel5;
      default:
        return shadowLevel2;
    }
  }

  /// 获取成功/错误/警告等状态颜色
  static Color getStatusColor(String status, {bool isDark = false}) {
    switch (status.toLowerCase()) {
      case 'success':
        return successColor;
      case 'warning':
        return warningColor;
      case 'error':
        return errorColor;
      case 'info':
        return infoColor;
      default:
        return isDark ? darkTextPrimary : lightTextPrimary;
    }
  }
}
