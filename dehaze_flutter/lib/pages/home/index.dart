import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../theme/app_theme.dart';
import 'algorithm_section.dart';
import 'cta_section.dart';
import 'hero_section.dart';
import 'showcase_section.dart';
import 'tools_grid_section.dart';
import 'workflow_section.dart';

// 首页主组件
class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 统一的区块间距
    final sectionSpacing = AppTheme.spacingXL * 1.5;

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          // Hero Section - 英雄区域
          const SliverToBoxAdapter(child: HeroSection()),

          // 区块间距
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),

          // 效果展示区
          const SliverToBoxAdapter(child: ShowcaseSection()),

          // 区块间距
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),

          // 核心功能区 - 包含工作流程和工具网格
          const SliverToBoxAdapter(child: FeaturesSection()),

          // 区块间距
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),

          // 算法优势区
          const SliverToBoxAdapter(child: AlgorithmSection()),

          // 区块间距
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),

          // 最终CTA区域
          const SliverToBoxAdapter(child: CTASection()),

          // 底部安全区域
          SliverToBoxAdapter(
            child: SizedBox(
              height: MediaQuery.of(context).padding.bottom + AppTheme.spacingXL,
            ),
          ),
        ],
      ),
    );
  }
}

/// 核心功能区组合组件
///
/// 将工作流程和工具网格组合在一起
class FeaturesSection extends StatelessWidget {
  const FeaturesSection({super.key});

  @override
  Widget build(BuildContext context) => Column(
    children: [
      // 工作流程区域
      WorkflowSection(),
      SizedBox(height: AppTheme.spacingXL),

      // 工具网格区域
      ToolsGridSection(),
    ],
  );
}
