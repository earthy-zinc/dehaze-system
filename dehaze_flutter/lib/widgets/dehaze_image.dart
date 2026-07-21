import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// 跨平台图片组件
///
/// 统一处理三种图片来源，避免在 Web 端使用 dart:io 的 [Image.file]：
/// 1. [bytes] 非空 → Image.memory（本地选择/拍摄的图片，全平台可用）
/// 2. [url] 为 http(s) 网络地址 → Image.network
/// 3. 其余情况（Web 端无法访问的本地路径）→ 占位图
///
/// 这是为了解决 Flutter Web 不支持 dart:io File 的问题：
/// 本地图片在选择时即读取为字节流，渲染时统一走 Image.memory。
class DehazeImage extends StatelessWidget {
  const DehazeImage({
    super.key,
    this.bytes,
    this.url,
    this.fit = BoxFit.cover,
    this.width,
    this.height,
    this.borderRadius,
    this.placeholderIcon = Icons.image_outlined,
    this.errorIcon = Icons.broken_image_outlined,
  });

  /// 图片字节流（本地图片的跨平台表示）
  final Uint8List? bytes;

  /// 图片 URL（网络地址或本地路径）
  final String? url;

  final BoxFit fit;
  final double? width;
  final double? height;
  final BorderRadius? borderRadius;
  final IconData placeholderIcon;
  final IconData errorIcon;

  bool get _isNetworkUrl {
    final u = url;
    return u != null && (u.startsWith('http://') || u.startsWith('https://'));
  }

  @override
  Widget build(BuildContext context) {
    Widget image;

    if (bytes != null && bytes!.isNotEmpty) {
      // 字节流渲染（全平台可用，Web 端本地图片的唯一可靠方式）
      image = Image.memory(
        bytes!,
        fit: fit,
        width: width,
        height: height,
        gaplessPlayback: true,
        errorBuilder: (_, _, _) => _buildError(context),
      );
    } else if (_isNetworkUrl) {
      // 网络图片
      image = Image.network(
        url!,
        fit: fit,
        width: width,
        height: height,
        gaplessPlayback: true,
        loadingBuilder: (context, child, progress) {
          if (progress == null) return child;
          return _buildLoading(context, progress);
        },
        errorBuilder: (_, _, _) => _buildError(context),
      );
    } else {
      // Web 端无法访问本地文件路径，显示占位图
      image = _buildPlaceholder(context);
    }

    if (borderRadius != null) {
      return ClipRRect(borderRadius: borderRadius!, child: image);
    }
    return image;
  }

  Widget _buildLoading(BuildContext context, ImageChunkEvent progress) {
    final expected = progress.expectedTotalBytes;
    final value = expected != null && expected > 0
        ? progress.cumulativeBytesLoaded / expected
        : null;
    return Center(
      child: SizedBox(
        width: 24,
        height: 24,
        child: CircularProgressIndicator(strokeWidth: 2, value: value),
      ),
    );
  }

  Widget _buildPlaceholder(BuildContext context) => Container(
        width: width,
        height: height,
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        child: Center(
          child: Icon(
            placeholderIcon,
            size: 40,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );

  Widget _buildError(BuildContext context) => Container(
        width: width,
        height: height,
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        child: Center(
          child: Icon(
            errorIcon,
            size: 40,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );
}
