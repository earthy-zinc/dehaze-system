import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';

class DehazePage extends StatefulWidget {
  const DehazePage({super.key});

  @override
  State<DehazePage> createState() => _DehazePageState();
}

class _DehazePageState extends State<DehazePage> {
  int _currentStep = 0;

  static const _steps = [
    _StepInfo(
      icon: Icons.cloud_upload_outlined,
      title: '上传图像',
      description: '选择需要去雾的图像',
    ),
    _StepInfo(
      icon: Icons.psychology_outlined,
      title: '选择算法',
      description: '智能推荐或手动选择',
    ),
    _StepInfo(
      icon: Icons.tune_outlined,
      title: '调节参数',
      description: '自定义处理参数',
    ),
    _StepInfo(
      icon: Icons.auto_fix_high,
      title: '开始处理',
      description: '执行去雾处理',
    ),
    _StepInfo(
      icon: Icons.compare_outlined,
      title: '效果对比',
      description: '查看处理结果',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final isWide = ResponsiveUtils.isWideScreen(context);

    return Scaffold(
      body: isWide ? _buildWideLayout(context) : _buildMobileLayout(context),
    );
  }

  Widget _buildMobileLayout(BuildContext context) {
    final theme = Theme.of(context);

    return Column(
      children: [
        Container(
          padding: EdgeInsets.all(AppTheme.spacingM),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            border: Border(
              bottom: BorderSide(color: theme.dividerColor),
            ),
          ),
          child: SafeArea(
            bottom: false,
            child: Row(
              children: [
                for (int i = 0; i < _steps.length; i++) ...[
                  if (i > 0)
                    Expanded(
                      child: Container(
                        height: 2,
                        color: i <= _currentStep
                            ? AppTheme.brandBlue
                            : theme.dividerColor,
                      ),
                    ),
                  _StepDot(
                    index: i,
                    isActive: i == _currentStep,
                    isCompleted: i < _currentStep,
                    onTap: () => setState(() => _currentStep = i),
                  ),
                ],
              ],
            ),
          ),
        ),
        SizedBox(height: AppTheme.spacingS),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
          child: Text(
            _steps[_currentStep].title,
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        SizedBox(height: AppTheme.spacingXS),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
          child: Text(
            _steps[_currentStep].description,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
        SizedBox(height: AppTheme.spacingL),
        Expanded(child: _buildStepContent(context)),
        _buildBottomActions(context),
      ],
    );
  }

  Widget _buildWideLayout(BuildContext context) {
    final theme = Theme.of(context);

    return Row(
      children: [
        Container(
          width: 240,
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest,
            border: Border(
              right: BorderSide(color: theme.dividerColor),
            ),
          ),
          child: SafeArea(
            child: ListView(
              padding: EdgeInsets.symmetric(vertical: AppTheme.spacingM),
              children: [
                Padding(
                  padding: EdgeInsets.symmetric(
                    horizontal: AppTheme.spacingM,
                    vertical: AppTheme.spacingS,
                  ),
                  child: Text(
                    '去雾流程',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                SizedBox(height: AppTheme.spacingS),
                for (int i = 0; i < _steps.length; i++)
                  _WideStepItem(
                    index: i,
                    info: _steps[i],
                    isActive: i == _currentStep,
                    isCompleted: i < _currentStep,
                    onTap: () => setState(() => _currentStep = i),
                  ),
              ],
            ),
          ),
        ),
        Expanded(
          child: Column(
            children: [
              Expanded(child: _buildStepContent(context)),
              _buildBottomActions(context),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildStepContent(BuildContext context) {
    final theme = Theme.of(context);

    switch (_currentStep) {
      case 0:
        return _buildUploadStep(context, theme);
      case 1:
        return _buildAlgorithmStep(context, theme);
      case 2:
        return _buildParamsStep(context, theme);
      case 3:
        return _buildProcessingStep(context, theme);
      case 4:
        return _buildComparisonStep(context, theme);
      default:
        return _buildUploadStep(context, theme);
    }
  }

  Widget _buildUploadStep(BuildContext context, ThemeData theme) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingXL),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: AppTheme.brandBlue.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(AppTheme.radiusXL),
                border: Border.all(
                  color: AppTheme.brandBlue.withValues(alpha: 0.2),
                  width: 2,
                  style: BorderStyle.solid,
                ),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.cloud_upload_outlined, size: 48, color: AppTheme.brandBlue),
                  SizedBox(height: AppTheme.spacingS),
                  Text(
                    '点击或拖拽上传',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: AppTheme.brandBlue,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: AppTheme.spacingL),
            Text(
              '支持 JPG、PNG、BMP 格式',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            SizedBox(height: AppTheme.spacingM),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildUploadMethod(Icons.camera_alt_outlined, '拍照', theme),
                SizedBox(width: AppTheme.spacingL),
                _buildUploadMethod(Icons.photo_library_outlined, '相册', theme),
                SizedBox(width: AppTheme.spacingL),
                _buildUploadMethod(Icons.collections_outlined, '样例', theme),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUploadMethod(IconData icon, String label, ThemeData theme) {
    return Column(
      children: [
        Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(AppTheme.radiusM),
          ),
          child: Icon(icon, color: theme.colorScheme.onSurfaceVariant, size: 24),
        ),
        SizedBox(height: AppTheme.spacingXS),
        Text(
          label,
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }

  Widget _buildAlgorithmStep(BuildContext context, ThemeData theme) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingXL),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.psychology_outlined, size: 80, color: AppTheme.techGreen),
            SizedBox(height: AppTheme.spacingL),
            Text(
              '选择去雾算法',
              style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600),
            ),
            SizedBox(height: AppTheme.spacingM),
            Text(
              '系统将根据图像特征智能推荐最适合的算法',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: AppTheme.spacingL),
            FilledButton.icon(
              onPressed: () => context.go(AppRouterConfig.algorithmSelect),
              icon: Icon(Icons.arrow_forward, size: 18),
              label: Text('进入算法选择'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildParamsStep(BuildContext context, ThemeData theme) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingXL),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.tune_outlined, size: 80, color: AppTheme.indigo),
            SizedBox(height: AppTheme.spacingL),
            Text(
              '调节处理参数',
              style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600),
            ),
            SizedBox(height: AppTheme.spacingM),
            Text(
              '自定义强度、亮度、对比度等参数',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: AppTheme.spacingL),
            Text(
              '参数调节将在算法选择后可用',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProcessingStep(BuildContext context, ThemeData theme) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingXL),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 80,
              height: 80,
              child: CircularProgressIndicator(strokeWidth: 3),
            ),
            SizedBox(height: AppTheme.spacingL),
            Text(
              '正在处理中...',
              style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600),
            ),
            SizedBox(height: AppTheme.spacingM),
            Text(
              'AI 算法正在处理您的图像',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            SizedBox(height: AppTheme.spacingL),
            FilledButton.icon(
              onPressed: () => context.go(AppRouterConfig.processing),
              icon: Icon(Icons.play_arrow, size: 18),
              label: Text('进入处理页面'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildComparisonStep(BuildContext context, ThemeData theme) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingXL),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.compare_outlined, size: 80, color: AppTheme.successColor),
            SizedBox(height: AppTheme.spacingL),
            Text(
              '查看对比效果',
              style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600),
            ),
            SizedBox(height: AppTheme.spacingM),
            Text(
              '支持并排、重叠、放大镜等多种对比方式',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: AppTheme.spacingL),
            FilledButton.icon(
              onPressed: () => context.go(AppRouterConfig.sideBySide),
              icon: Icon(Icons.arrow_forward, size: 18),
              label: Text('进入对比页'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomActions(BuildContext context) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingM),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            if (_currentStep > 0)
              OutlinedButton.icon(
                onPressed: () => setState(() => _currentStep--),
                icon: Icon(Icons.arrow_back, size: 18),
                label: Text('上一步'),
              )
            else
              SizedBox(width: 120),
            FilledButton.icon(
              onPressed: () {
                if (_currentStep < _steps.length - 1) {
                  setState(() => _currentStep++);
                } else {
                  context.go(AppRouterConfig.sideBySide);
                }
              },
              icon: Icon(
                _currentStep < _steps.length - 1
                    ? Icons.arrow_forward
                    : Icons.check,
                size: 18,
              ),
              label: Text(
                _currentStep < _steps.length - 1 ? '下一步' : '完成',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StepInfo {
  const _StepInfo({
    required this.icon,
    required this.title,
    required this.description,
  });

  final IconData icon;
  final String title;
  final String description;
}

class _StepDot extends StatelessWidget {
  const _StepDot({
    required this.index,
    required this.isActive,
    required this.isCompleted,
    required this.onTap,
  });

  final int index;
  final bool isActive;
  final bool isCompleted;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 28,
        height: 28,
        decoration: BoxDecoration(
          color: isActive || isCompleted ? AppTheme.brandBlue : Colors.transparent,
          shape: BoxShape.circle,
          border: Border.all(
            color: isActive || isCompleted
                ? AppTheme.brandBlue
                : Theme.of(context).dividerColor,
            width: 2,
          ),
        ),
        child: Center(
          child: isCompleted
              ? Icon(Icons.check, color: Colors.white, size: 14)
              : Text(
                  '${index + 1}',
                  style: TextStyle(
                    color: isActive ? Colors.white : Theme.of(context).colorScheme.onSurfaceVariant,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
        ),
      ),
    );
  }
}

class _WideStepItem extends StatelessWidget {
  const _WideStepItem({
    required this.index,
    required this.info,
    required this.isActive,
    required this.isCompleted,
    required this.onTap,
  });

  final int index;
  final _StepInfo info;
  final bool isActive;
  final bool isCompleted;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = isActive || isCompleted
        ? AppTheme.brandBlue
        : theme.colorScheme.onSurfaceVariant;

    return Padding(
      padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingS, vertical: 2),
      child: Material(
        color: isActive
            ? AppTheme.brandBlue.withValues(alpha: 0.08)
            : Colors.transparent,
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(AppTheme.radiusM),
          child: Padding(
            padding: EdgeInsets.symmetric(
              horizontal: AppTheme.spacingM,
              vertical: AppTheme.spacingS,
            ),
            child: Row(
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: isCompleted
                        ? AppTheme.successColor
                        : isActive
                            ? AppTheme.brandBlue
                            : theme.colorScheme.surfaceContainerHighest,
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: isCompleted
                        ? Icon(Icons.check, color: Colors.white, size: 16)
                        : Text(
                            '${index + 1}',
                            style: TextStyle(
                              color: isActive ? Colors.white : color,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                  ),
                ),
                SizedBox(width: AppTheme.spacingM),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        info.title,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                          color: color,
                        ),
                      ),
                      Text(
                        info.description,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
                if (isActive)
                  Icon(Icons.arrow_forward_ios, size: 14, color: AppTheme.brandBlue),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
