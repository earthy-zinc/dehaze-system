import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../theme/app_theme.dart';
import '../../router/config.dart';

/// 工具网格区域组件
///
/// 展示各种对比和分析工具，支持响应式布局
class ToolsGridSection extends StatelessWidget {
  const ToolsGridSection({super.key});

  @override
  Widget build(BuildContext context) {
    final tools = [
      _ToolData(
        icon: Icons.view_column,
        title: '并排对比',
        desc: '多图并排展示，支持2-4张图片同屏对比',
        route: AppRouterConfig.sideBySide,
      ),
      _ToolData(
        icon: Icons.layers,
        title: '重叠对比',
        desc: '拖动分割线实时对比，支持横向和纵向模式',
        route: AppRouterConfig.overlay,
      ),
      _ToolData(
        icon: Icons.search,
        title: '放大镜',
        desc: '局部细节放大查看，精确对比图像质量',
        route: AppRouterConfig.magnifier,
      ),
      _ToolData(
        icon: Icons.tune,
        title: '滤镜调节',
        desc: '实时调节亮度、对比度、饱和度等参数',
        route: AppRouterConfig.filter,
      ),
      _ToolData(
        icon: Icons.analytics,
        title: '指标评估',
        desc: 'SSIM、PSNR等专业指标定量分析',
        route: AppRouterConfig.metrics,
      ),
      _ToolData(
        icon: Icons.storage,
        title: '数据集管理',
        desc: '浏览和管理多个专业去雾数据集',
        route: AppRouterConfig.dataset,
      ),
    ];

    return Padding(
      padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
      child: LayoutBuilder(
        builder: (context, constraints) {
          // 响应式列数计算
          final crossAxisCount = _getCrossAxisCount(constraints.maxWidth);
          final childAspectRatio = _getAspectRatio(constraints.maxWidth);

          return GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: crossAxisCount,
              crossAxisSpacing: AppTheme.spacingM,
              mainAxisSpacing: AppTheme.spacingM,
              childAspectRatio: childAspectRatio,
            ),
            itemCount: tools.length,
            itemBuilder: (context, index) => _ToolCard(data: tools[index]),
          );
        },
      ),
    );
  }

  /// 根据屏幕宽度获取列数
  int _getCrossAxisCount(double width) {
    if (width > 1200) return 3; // 大屏幕3列，让卡片更大
    if (width > 800) return 3;
    if (width > 480) return 2;
    return 2;
  }

  /// 根据屏幕宽度获取宽高比
  double _getAspectRatio(double width) {
    if (width > 1200) return 1.2; // 大屏幕
    if (width > 800) return 1.1;  // 中等屏幕，降低宽高比增加高度
    if (width > 480) return 1.0;
    return 0.95;
  }
}

/// 工具数据模型
class _ToolData {
  const _ToolData({
    required this.icon,
    required this.title,
    required this.desc,
    required this.route,
  });

  final IconData icon;
  final String title;
  final String desc;
  final String route;
}

/// 单个工具卡片组件
class _ToolCard extends StatefulWidget {
  const _ToolCard({required this.data});

  final _ToolData data;

  @override
  State<_ToolCard> createState() => _ToolCardState();
}

class _ToolCardState extends State<_ToolCard> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        transform: Matrix4.identity()..setEntry(0, 0, _isHovered ? 1.02 : 1.0)..setEntry(1, 1, _isHovered ? 1.02 : 1.0),
        child: Card(
          elevation: _isHovered ? 8 : 2,
          shadowColor: AppTheme.brandBlue.withValues(alpha: _isHovered ? 0.3 : 0.1),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            side: BorderSide(
              color: _isHovered
                  ? AppTheme.brandBlue.withValues(alpha: 0.3)
                  : theme.dividerColor,
              width: _isHovered ? 2 : 1,
            ),
          ),
          child: InkWell(
            onTap: () => context.go(widget.data.route),
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            child: Padding(
              padding: EdgeInsets.all(AppTheme.spacingM),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // 图标容器 - 使用 Flexible 防止溢出
                  Flexible(
                    flex: 3,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      constraints: const BoxConstraints(
                        maxWidth: 52,
                        maxHeight: 52,
                      ),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: _isHovered
                              ? [AppTheme.brandBlue, AppTheme.brandBlueDark]
                              : AppTheme.toolCardGradient,
                        ),
                        borderRadius: BorderRadius.circular(AppTheme.radiusM),
                      ),
                      child: Center(
                        child: Icon(
                          widget.data.icon,
                          color: _isHovered ? Colors.white : AppTheme.brandBlue,
                          size: 24,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: AppTheme.spacingS),

                  // 标题
                  Flexible(
                    flex: 1,
                    child: Text(
                      widget.data.title,
                      style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: _isHovered ? AppTheme.brandBlue : null,
                      ),
                      textAlign: TextAlign.center,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  SizedBox(height: AppTheme.spacingXS),

                  // 描述
                  Flexible(
                    flex: 2,
                    child: Text(
                      widget.data.desc,
                      style: theme.textTheme.bodySmall?.copyWith(
                        height: 1.3,
                        fontSize: 11,
                        color: theme.textTheme.bodySmall?.color?.withValues(alpha: 0.8),
                      ),
                      textAlign: TextAlign.center,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),

                  // 底部操作提示 - 仅在悬停时显示
                  Flexible(
                    flex: 1,
                    child: AnimatedOpacity(
                      duration: const Duration(milliseconds: 200),
                      opacity: _isHovered ? 1.0 : 0.0,
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        height: _isHovered ? null : 0,
                        padding: EdgeInsets.symmetric(
                          horizontal: AppTheme.spacingS,
                          vertical: AppTheme.spacingXS,
                        ),
                        decoration: BoxDecoration(
                          color: AppTheme.brandBlue.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(AppTheme.radiusS),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              '立即体验',
                              style: theme.textTheme.labelSmall?.copyWith(
                                color: AppTheme.brandBlue,
                                fontWeight: FontWeight.w600,
                                fontSize: 10,
                              ),
                            ),
                            SizedBox(width: AppTheme.spacingXS),
                            Icon(
                              Icons.arrow_forward,
                              size: 12,
                              color: AppTheme.brandBlue,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
