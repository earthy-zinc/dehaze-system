import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../common/empty_state_widget.dart';
import '../common/error_widget.dart';
import '../common/loading_widget.dart';

class AppScaffold extends ConsumerWidget {

  const AppScaffold({
    required this.body, super.key,
    this.title,
    this.floatingActionButton,
    this.bottomNavigationBar,
    this.actions,
    this.showAppBar = true,
    this.isLoading = false,
    this.errorMessage,
    this.onRetry,
    this.emptyWidget,
    this.isEmpty = false,
  });
  final String? title;
  final Widget body;
  final Widget? floatingActionButton;
  final Widget? bottomNavigationBar;
  final List<Widget>? actions;
  final bool showAppBar;
  final bool isLoading;
  final String? errorMessage;
  final VoidCallback? onRetry;
  final Widget? emptyWidget;
  final bool isEmpty;

  @override
  Widget build(BuildContext context, WidgetRef ref) => Scaffold(
      appBar: showAppBar
          ? AppBar(
              title: title != null ? Text(title!) : null,
              actions: actions,
              elevation: 0,
              backgroundColor: Theme.of(context).scaffoldBackgroundColor,
              foregroundColor: Theme.of(context).textTheme.titleLarge?.color,
            )
          : null,
      body: _buildBody(),
      floatingActionButton: floatingActionButton,
      bottomNavigationBar: bottomNavigationBar,
    );

  Widget _buildBody() {
    if (isLoading) {
      return const LoadingWidget(message: '加载中...', size: 24);
    }

    if (errorMessage != null) {
      return AppErrorWidget(
        message: '加载失败',
        details: errorMessage,
        onRetry: onRetry,
      );
    }

    if (isEmpty) {
      return emptyWidget ??
          const EmptyStateWidget(title: '暂无数据', subtitle: '当前没有任何内容');
    }

    return body;
  }
}
