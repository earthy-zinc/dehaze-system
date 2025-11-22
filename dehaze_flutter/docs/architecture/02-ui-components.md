# UI组件与交互设计

**文档版本**: v1.0
**最后更新**: 2025-11-22
**项目名称**: dehaze_flutter
**参考文档**: [设计系统](../design/01-design-system.md)、[业务组件](../design/05-business-components.md)

---

## 📋 概述

本文档详细描述了Flutter图像去雾系统的UI组件设计和交互规范，基于[设计系统文档](../design/01-design-system.md)和[业务组件设计](../design/05-business-components.md)，专注于Flutter平台的实现细节和用户交互优化。

---

## 🏗️ 组件架构设计

### 组件分层架构

```
应用组件层 (Application Components)
    ↓
业务组件层 (Business Components)
    ↓
通用组件层 (Common Components)
    ↓
基础组件层 (Base Components)
```

### 组件分类体系

| 组件类型 | 用途 | 示例 | 复用性 |
|---------|------|------|--------|
| **基础组件** | 构建块级UI元素 | AppButton、AppCard | 高 |
| **布局组件** | 页面布局和容器 | AppScaffold、AppContainer | 中 |
| **表单组件** | 用户输入控件 | AppTextField、AppSlider | 高 |
| **业务组件** | 特定功能组件 | ImageUploader、AlgorithmSelector | 低 |
| **页面组件** | 完整页面实现 | HomePage、ProcessingPage | 低 |

---

## 🎨 通用组件设计

### 基础按钮组件 (AppButton)

#### 设计规范
基于[设计系统中的按钮规范](../design/01-design-system.md#5-按钮组件)，实现Flutter版本的按钮组件。

#### 组件实现
```dart
enum AppButtonType {
  primary,     // 主要按钮
  secondary,   // 次要按钮
  outline,     // 轮廓按钮
  text,        // 文本按钮
  icon,        // 图标按钮
}

enum AppButtonSize {
  small,       // 40px高度
  medium,      // 48px高度
  large,       // 56px高度
}

class AppButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final AppButtonType type;
  final AppButtonSize size;
  final IconData? icon;
  final bool isLoading;
  final bool fullWidth;

  const AppButton({
    Key? key,
    required this.text,
    this.onPressed,
    this.type = AppButtonType.primary,
    this.size = AppButtonSize.medium,
    this.icon,
    this.isLoading = false,
    this.fullWidth = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: fullWidth ? double.infinity : null,
      height: _getHeight(),
      child: ElevatedButton(
        onPressed: isLoading ? null : onPressed,
        style: _getButtonStyle(context),
        child: isLoading
            ? _buildLoadingIndicator()
            : _buildButtonContent(),
      ),
    );
  }

  double _getHeight() {
    switch (size) {
      case AppButtonSize.small:
        return 40;
      case AppButtonSize.medium:
        return 48;
      case AppButtonSize.large:
        return 56;
    }
  }

  ButtonStyle _getButtonStyle(BuildContext context) {
    final theme = Theme.of(context);

    switch (type) {
      case AppButtonType.primary:
        return ElevatedButton.styleFrom(
          backgroundColor: theme.primaryColor,
          foregroundColor: Colors.white,
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        );
      case AppButtonType.secondary:
        return ElevatedButton.styleFrom(
          backgroundColor: Colors.grey[100],
          foregroundColor: Colors.grey[700],
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        );
      case AppButtonType.outline:
        return OutlinedButton.styleFrom(
          foregroundColor: theme.primaryColor,
          side: BorderSide(color: theme.primaryColor),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        );
      case AppButtonType.text:
        return TextButton.styleFrom(
          foregroundColor: theme.primaryColor,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        );
      case AppButtonType.icon:
        return IconButton.styleFrom(
          foregroundColor: theme.primaryColor,
          backgroundColor: Colors.grey[100],
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        );
    }
  }

  Widget _buildButtonContent() {
    if (icon != null && type != AppButtonType.icon) {
      return Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: _getIconSize()),
          SizedBox(width: 8),
          Text(text, style: _getTextStyle()),
        ],
      );
    } else if (type == AppButtonType.icon) {
      return Icon(icon, size: _getIconSize());
    } else {
      return Text(text, style: _getTextStyle());
    }
  }

  Widget _buildLoadingIndicator() {
    return SizedBox(
      width: 20,
      height: 20,
      child: CircularProgressIndicator(
        strokeWidth: 2,
        valueColor: AlwaysStoppedAnimation<Color>(
          type == AppButtonType.primary ? Colors.white : Colors.blue,
        ),
      ),
    );
  }

  double _getIconSize() {
    switch (size) {
      case AppButtonSize.small:
        return 16;
      case AppButtonSize.medium:
        return 20;
      case AppButtonSize.large:
        return 24;
    }
  }

  TextStyle _getTextStyle() {
    switch (size) {
      case AppButtonSize.small:
        return TextStyle(fontSize: 14, fontWeight: FontWeight.w500);
      case AppButtonSize.medium:
        return TextStyle(fontSize: 16, fontWeight: FontWeight.w600);
      case AppButtonSize.large:
        return TextStyle(fontSize: 18, fontWeight: FontWeight.w600);
    }
  }
}
```

#### 使用示例
```dart
// 主要按钮
AppButton(
  text: '开始处理',
  onPressed: () => _startProcessing(),
  type: AppButtonType.primary,
  size: AppButtonSize.large,
  fullWidth: true,
)

// 带图标的按钮
AppButton(
  text: '上传图片',
  onPressed: () => _uploadImage(),
  icon: Icons.upload_file,
)

// 加载状态按钮
AppButton(
  text: '处理中...',
  isLoading: true,
  onPressed: null,
)
```

### 卡片组件 (AppCard)

#### 设计规范
基于[设计系统中的卡片规范](../design/01-design-system.md#5-3-卡片组件)，实现响应式卡片组件。

#### 组件实现
```dart
enum AppCardType {
  basic,        // 基础卡片
  elevated,     // 提升卡片
  outlined,     // 轮廓卡片
  featured,     // 特色卡片
}

class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final VoidCallback? onTap;
  final AppCardType type;
  final Color? backgroundColor;
  final double? elevation;
  final BorderRadius? borderRadius;

  const AppCard({
    Key? key,
    required this.child,
    this.padding,
    this.margin,
    this.onTap,
    this.type = AppCardType.basic,
    this.backgroundColor,
    this.elevation,
    this.borderRadius,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final card = Container(
      margin: margin ?? const EdgeInsets.all(16),
      decoration: _getCardDecoration(context),
      child: InkWell(
        onTap: onTap,
        borderRadius: borderRadius ?? BorderRadius.circular(16),
        child: Padding(
          padding: padding ?? const EdgeInsets.all(16),
          child: child,
        ),
      ),
    );

    if (onTap != null) {
      return Material(
        color: Colors.transparent,
        child: card,
      );
    }

    return card;
  }

  BoxDecoration _getCardDecoration(BuildContext context) {
    final theme = Theme.of(context);

    switch (type) {
      case AppCardType.basic:
        return BoxDecoration(
          color: backgroundColor ?? Colors.white,
          borderRadius: borderRadius ?? BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.08),
              blurRadius: 8,
              offset: Offset(0, 2),
            ),
          ],
        );
      case AppCardType.elevated:
        return BoxDecoration(
          color: backgroundColor ?? Colors.white,
          borderRadius: borderRadius ?? BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.15),
              blurRadius: 12,
              offset: Offset(0, 4),
            ),
          ],
        );
      case AppCardType.outlined:
        return BoxDecoration(
          color: backgroundColor ?? Colors.white,
          borderRadius: borderRadius ?? BorderRadius.circular(16),
          border: Border.all(
            color: Colors.grey[300]!,
            width: 1,
          ),
        );
      case AppCardType.featured:
        return BoxDecoration(
          gradient: LinearGradient(
            colors: [
              theme.primaryColor,
              theme.primaryColor.withOpacity(0.8),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: borderRadius ?? BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: theme.primaryColor.withOpacity(0.3),
              blurRadius: 16,
              offset: Offset(0, 4),
            ),
          ],
        );
    }
  }
}
```

#### 使用示例
```dart
// 基础卡片
AppCard(
  child: Column(
    children: [
      Text('卡片标题'),
      Text('卡片内容'),
    ],
  ),
)

// 可点击卡片
AppCard(
  onTap: () => _navigateToDetail(),
  type: AppCardType.elevated,
  child: ListTile(
    title: Text('点击查看详情'),
    trailing: Icon(Icons.arrow_forward_ios),
  ),
)

// 特色卡片
AppCard(
  type: AppCardType.featured,
  child: Column(
    children: [
      Icon(Icons.star, color: Colors.white, size: 32),
      SizedBox(height: 8),
      Text('特色功能', style: TextStyle(color: Colors.white)),
    ],
  ),
)
```

### 输入框组件 (AppTextField)

#### 组件实现
```dart
enum AppTextFieldType {
  text,         // 文本输入
  password,     // 密码输入
  email,        // 邮箱输入
  number,       // 数字输入
  search,       // 搜索输入
  multiline,    // 多行输入
}

class AppTextField extends StatefulWidget {
  final String? label;
  final String? hint;
  final String? errorText;
  final String? helperText;
  final IconData? prefixIcon;
  final IconData? suffixIcon;
  final VoidCallback? onSuffixIconTap;
  final ValueChanged<String>? onChanged;
  final VoidCallback? onTap;
  final TextEditingController? controller;
  final FocusNode? focusNode;
  final bool obscureText;
  final bool readOnly;
  final int? maxLines;
  final TextInputType? keyboardType;
  final List<TextInputFormatter>? inputFormatters;
  final String? Function(String?)? validator;

  const AppTextField({
    Key? key,
    this.label,
    this.hint,
    this.errorText,
    this.helperText,
    this.prefixIcon,
    this.suffixIcon,
    this.onSuffixIconTap,
    this.onChanged,
    this.onTap,
    this.controller,
    this.focusNode,
    this.obscureText = false,
    this.readOnly = false,
    this.maxLines = 1,
    this.keyboardType,
    this.inputFormatters,
    this.validator,
  }) : super(key: key);

  @override
  State<AppTextField> createState() => _AppTextFieldState();
}

class _AppTextFieldState extends State<AppTextField> {
  bool _hasFocus = false;
  late FocusNode _focusNode;

  @override
  void initState() {
    super.initState();
    _focusNode = widget.focusNode ?? FocusNode();
    _focusNode.addListener(_onFocusChange);
  }

  @override
  void dispose() {
    if (widget.focusNode == null) {
      _focusNode.dispose();
    }
    super.dispose();
  }

  void _onFocusChange() {
    setState(() {
      _hasFocus = _focusNode.hasFocus;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (widget.label != null) ...[
          Text(
            widget.label!,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: _hasFocus ? Theme.of(context).primaryColor : Colors.grey[700],
            ),
          ),
          SizedBox(height: 8),
        ],
        Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: widget.errorText != null
                  ? Colors.red
                  : _hasFocus
                      ? Theme.of(context).primaryColor
                      : Colors.grey[300]!,
              width: 2,
            ),
            boxShadow: [
              if (_hasFocus)
                BoxShadow(
                  color: Theme.of(context).primaryColor.withOpacity(0.2),
                  blurRadius: 8,
                  offset: Offset(0, 2),
                ),
            ],
          ),
          child: TextField(
            controller: widget.controller,
            focusNode: _focusNode,
            obscureText: widget.obscureText,
            readOnly: widget.readOnly,
            maxLines: widget.maxLines,
            keyboardType: widget.keyboardType,
            inputFormatters: widget.inputFormatters,
            onChanged: widget.onChanged,
            onTap: widget.onTap,
            decoration: InputDecoration(
              hintText: widget.hint,
              prefixIcon: widget.prefixIcon != null
                  ? Icon(widget.prefixIcon, color: Colors.grey[600])
                  : null,
              suffixIcon: widget.suffixIcon != null
                  ? IconButton(
                      icon: Icon(widget.suffixIcon, color: Colors.grey[600]),
                      onPressed: widget.onSuffixIconTap,
                    )
                  : null,
              border: InputBorder.none,
              contentPadding: EdgeInsets.all(16),
              hintStyle: TextStyle(color: Colors.grey[400]),
            ),
          ),
        ),
        if (widget.errorText != null) ...[
          SizedBox(height: 4),
          Text(
            widget.errorText!,
            style: TextStyle(
              color: Colors.red,
              fontSize: 12,
            ),
          ),
        ],
        if (widget.helperText != null) ...[
          SizedBox(height: 4),
          Text(
            widget.helperText!,
            style: TextStyle(
              color: Colors.grey[600],
              fontSize: 12,
            ),
          ),
        ],
      ],
    );
  }
}
```

### 滑块组件 (AppSlider)

#### 组件实现
```dart
class AppSlider extends StatelessWidget {
  final double value;
  final ValueChanged<double>? onChanged;
  final double min;
  final double max;
  final int? divisions;
  final String? label;
  final bool showValue;
  final Color? activeColor;
  final Color? inactiveColor;

  const AppSlider({
    Key? key,
    required this.value,
    this.onChanged,
    this.min = 0.0,
    this.max = 1.0,
    this.divisions,
    this.label,
    this.showValue = true,
    this.activeColor,
    this.inactiveColor,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (label != null) ...[
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label!,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[700],
                ),
              ),
              if (showValue)
                Text(
                  '${(value * 100).toInt()}%',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Theme.of(context).primaryColor,
                  ),
                ),
            ],
          ),
          SizedBox(height: 12),
        ],
        Container(
          child: SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: activeColor ?? Theme.of(context).primaryColor,
              inactiveTrackColor: inactiveColor ?? Colors.grey[300],
              thumbColor: activeColor ?? Theme.of(context).primaryColor,
              thumbShape: RoundSliderThumbShape(
                enabledThumbRadius: 12,
              ),
              overlayShape: RoundSliderOverlayShape(
                overlayRadius: 20,
              ),
              trackHeight: 6,
            ),
            child: Slider(
              value: value,
              min: min,
              max: max,
              divisions: divisions,
              onChanged: onChanged,
            ),
          ),
        ),
      ],
    );
  }
}
```

---

## 🎯 业务组件设计

### 图像输入组件 (ImageInputWidget)

#### 组件职责
- 提供多种图像输入方式
- 文件格式验证和处理
- 缩略图生成和预览
- 批量选择和管理

#### 组件实现
```dart
enum ImageInputMethod {
  gallery,      // 相册选择
  camera,       // 拍照
  file,         // 文件选择
  sample,       // 样例图片
  history,      // 历史记录
}

class ImageInputWidget extends StatefulWidget {
  final List<ImageFile> selectedImages;
  final ValueChanged<List<ImageFile>>? onImagesChanged;
  final int maxImages;
  final List<String> supportedFormats;
  final int maxFileSizeMB;

  const ImageInputWidget({
    Key? key,
    required this.selectedImages,
    this.onImagesChanged,
    this.maxImages = 5,
    this.supportedFormats = const ['JPG', 'PNG', 'WEBP', 'HEIC'],
    this.maxFileSizeMB = 20,
  }) : super(key: key);

  @override
  State<ImageInputWidget> createState() => _ImageInputWidgetState();
}

class _ImageInputWidgetState extends State<ImageInputWidget> {
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 输入方式选择
        _buildInputMethodGrid(),
        SizedBox(height: 24),
        // 已选择图片预览
        if (widget.selectedImages.isNotEmpty) ...[
          Text(
            '已选择图片 (${widget.selectedImages.length}/${widget.maxImages})',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          SizedBox(height: 16),
          _buildSelectedImagesGrid(),
        ],
      ],
    );
  }

  Widget _buildInputMethodGrid() {
    return GridView.count(
      shrinkWrap: true,
      physics: NeverScrollableScrollPhysics(),
      crossAxisCount: 2,
      mainAxisSpacing: 16,
      crossAxisSpacing: 16,
      childAspectRatio: 1.2,
      children: ImageInputMethod.values.map((method) {
        return _buildInputMethodCard(method);
      }).toList(),
    );
  }

  Widget _buildInputMethodCard(ImageInputMethod method) {
    final methodInfo = _getMethodInfo(method);

    return AppCard(
      onTap: () => _handleInputMethod(method),
      type: AppCardType.elevated,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            methodInfo['icon'],
            size: 48,
            color: Theme.of(context).primaryColor,
          ),
          SizedBox(height: 12),
          Text(
            methodInfo['title'],
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 8),
          Text(
            methodInfo['description'],
            style: TextStyle(
              fontSize: 12,
              color: Colors.grey[600],
            ),
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  Map<String, dynamic> _getMethodInfo(ImageInputMethod method) {
    switch (method) {
      case ImageInputMethod.gallery:
        return {
          'icon': Icons.photo_library,
          'title': '相册选择',
          'description': '从手机相册选择图片',
        };
      case ImageInputMethod.camera:
        return {
          'icon': Icons.camera_alt,
          'title': '拍照',
          'description': '使用相机拍摄新图片',
        };
      case ImageInputMethod.file:
        return {
          'icon': Icons.folder_open,
          'title': '文件选择',
          'description': '从文件系统选择图片',
        };
      case ImageInputMethod.sample:
        return {
          'icon': Icons.image,
          'title': '样例图片',
          'description': '使用预设的样例图片',
        };
      case ImageInputMethod.history:
        return {
          'icon': Icons.history,
          'title': '历史记录',
          'description': '从历史记录中选择',
        };
    }
  }

  Widget _buildSelectedImagesGrid() {
    return GridView.builder(
      shrinkWrap: true,
      physics: NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1,
      ),
      itemCount: widget.selectedImages.length + (widget.selectedImages.length < widget.maxImages ? 1 : 0),
      itemBuilder: (context, index) {
        if (index == widget.selectedImages.length) {
          return _buildAddMoreButton();
        }
        return _buildImageCard(widget.selectedImages[index], index);
      },
    );
  }

  Widget _buildImageCard(ImageFile imageFile, int index) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey[300]!),
        image: DecorationImage(
          image: FileImage(imageFile.file),
          fit: BoxFit.cover,
        ),
      ),
      child: Stack(
        children: [
          // 删除按钮
          Positioned(
            top: 4,
            right: 4,
            child: GestureDetector(
              onTap: () => _removeImage(index),
              child: Container(
                width: 24,
                height: 24,
                decoration: BoxDecoration(
                  color: Colors.red,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.close,
                  color: Colors.white,
                  size: 16,
                ),
              ),
            ),
          ),
          // 图片信息
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.all(4),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(12),
                  bottomRight: Radius.circular(12),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    imageFile.name,
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  Text(
                    '${(imageFile.size / 1024 / 1024).toStringAsFixed(1)} MB',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 8,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAddMoreButton() {
    return GestureDetector(
      onTap: () => _showInputMethodBottomSheet(),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: Colors.grey[300]!,
            style: BorderStyle.solid,
            width: 2,
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.add,
              size: 32,
              color: Colors.grey[600],
            ),
            SizedBox(height: 8),
            Text(
              '添加更多',
              style: TextStyle(
                color: Colors.grey[600],
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _handleInputMethod(ImageInputMethod method) async {
    switch (method) {
      case ImageInputMethod.gallery:
        await _pickFromGallery();
        break;
      case ImageInputMethod.camera:
        await _pickFromCamera();
        break;
      case ImageInputMethod.file:
        await _pickFromFile();
        break;
      case ImageInputMethod.sample:
        _navigateToSampleGallery();
        break;
      case ImageInputMethod.history:
        _navigateToHistory();
        break;
    }
  }

  Future<void> _pickFromGallery() async {
    try {
      final ImagePicker picker = ImagePicker();
      final List<XFile> images = await picker.pickMultiImage();

      if (images.isNotEmpty) {
        final validImages = <ImageFile>[];
        for (final image in images) {
          if (await _validateImage(image)) {
            validImages.add(ImageFile.fromXFile(image));
          }
        }

        if (validImages.isNotEmpty) {
          _addImages(validImages);
        }
      }
    } catch (e) {
      _showErrorDialog('选择图片失败: $e');
    }
  }

  Future<bool> _validateImage(XFile image) async {
    // 检查文件格式
    final extension = image.path.split('.').last.toUpperCase();
    if (!widget.supportedFormats.contains(extension)) {
      _showErrorDialog('不支持的图片格式: $extension');
      return false;
    }

    // 检查文件大小
    final fileSize = await image.length();
    if (fileSize > widget.maxFileSizeMB * 1024 * 1024) {
      _showErrorDialog('图片大小超过 ${widget.maxFileSizeMB}MB 限制');
      return false;
    }

    return true;
  }

  void _addImages(List<ImageFile> images) {
    final newImages = [...widget.selectedImages, ...images];
    if (newImages.length <= widget.maxImages) {
      widget.onImagesChanged?.call(newImages);
    } else {
      final allowedImages = newImages.take(widget.maxImages).toList();
      widget.onImagesChanged?.call(allowedImages);
      _showErrorDialog('最多只能选择 ${widget.maxImages} 张图片');
    }
  }

  void _removeImage(int index) {
    final newImages = [...widget.selectedImages];
    newImages.removeAt(index);
    widget.onImagesChanged?.call(newImages);
  }

  void _showInputMethodBottomSheet() {
    showModalBottomSheet(
      context: context,
      builder: (context) => _buildInputMethodBottomSheet(),
    );
  }

  Widget _buildInputMethodBottomSheet() {
    return Container(
      padding: EdgeInsets.all(16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '选择输入方式',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 24),
          ...ImageInputMethod.values.map((method) {
            final info = _getMethodInfo(method);
            return ListTile(
              leading: Icon(info['icon']),
              title: Text(info['title']),
              subtitle: Text(info['description']),
              onTap: () {
                Navigator.pop(context);
                _handleInputMethod(method);
              },
            );
          }).toList(),
        ],
      ),
    );
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('错误'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('确定'),
          ),
        ],
      ),
    );
  }

  void _navigateToSampleGallery() {
    Navigator.pushNamed(context, '/sample_gallery');
  }

  void _navigateToHistory() {
    Navigator.pushNamed(context, '/history');
  }

  Future<void> _pickFromCamera() async {
    // 实现相机拍照逻辑
  }

  Future<void> _pickFromFile() async {
    // 实现文件选择逻辑
  }
}
```

### 算法选择组件 (AlgorithmSelectorWidget)

#### 组件职责
- 显示算法列表和推荐
- 提供搜索和筛选功能
- 展示算法详细信息
- 管理算法收藏

#### 组件实现
```dart
class AlgorithmSelectorWidget extends StatefulWidget {
  final Algorithm? selectedAlgorithm;
  final ValueChanged<Algorithm?>? onAlgorithmSelected;
  final ImageFile? imageFile;

  const AlgorithmSelectorWidget({
    Key? key,
    this.selectedAlgorithm,
    this.onAlgorithmSelected,
    this.imageFile,
  }) : super(key: key);

  @override
  State<AlgorithmSelectorWidget> createState() => _AlgorithmSelectorWidgetState();
}

class _AlgorithmSelectorWidgetState extends State<AlgorithmSelectorWidget> {
  List<Algorithm> _algorithms = [];
  List<Algorithm> _filteredAlgorithms = [];
  List<Algorithm> _recommendedAlgorithms = [];
  Set<String> _favoriteAlgorithms = {};
  String _searchQuery = '';
  AlgorithmFilter _filter = AlgorithmFilter();

  @override
  void initState() {
    super.initState();
    _loadAlgorithms();
    if (widget.imageFile != null) {
      _loadRecommendedAlgorithms();
    }
  }

  @override
  Widget build(BuildContext context) {
    return CustomScrollView(
      slivers: [
        // 推荐算法区域
        if (_recommendedAlgorithms.isNotEmpty)
          SliverToBoxAdapter(
            child: _buildRecommendedSection(),
          ),

        // 搜索栏
        SliverToBoxAdapter(
          child: _buildSearchBar(),
        ),

        // 筛选器
        SliverToBoxAdapter(
          child: _buildFilterBar(),
        ),

        // 算法列表
        SliverList(
          delegate: SliverChildBuilderDelegate(
            (context, index) => _buildAlgorithmCard(_filteredAlgorithms[index]),
            childCount: _filteredAlgorithms.length,
          ),
        ),
      ],
    );
  }

  Widget _buildRecommendedSection() {
    return Container(
      margin: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.auto_awesome, color: Colors.orange),
              SizedBox(width: 8),
              Text(
                '智能推荐',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          SizedBox(height: 16),
          Container(
            height: 180,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: _recommendedAlgorithms.length,
              itemBuilder: (context, index) {
                final algorithm = _recommendedAlgorithms[index];
                return _buildRecommendedCard(algorithm);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendedCard(Algorithm algorithm) {
    return Container(
      width: 280,
      margin: EdgeInsets.only(right: 16),
      child: AppCard(
        onTap: () => widget.onAlgorithmSelected?.call(algorithm),
        type: AppCardType.elevated,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  backgroundColor: Colors.blue.withOpacity(0.1),
                  child: Icon(Icons.psychology, color: Colors.blue),
                ),
                SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        algorithm.name,
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        algorithm.type.displayName,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: Icon(
                    _favoriteAlgorithms.contains(algorithm.id)
                        ? Icons.favorite
                        : Icons.favorite_border,
                    ),
                  onPressed: () => _toggleFavorite(algorithm.id),
                ),
              ],
            ),
            SizedBox(height: 12),
            Row(
              children: [
                _buildRatingStars(algorithm.rating),
                SizedBox(width: 8),
                Text(
                  algorithm.rating.toStringAsFixed(1),
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
              ],
            ),
            SizedBox(height: 8),
            Text(
              algorithm.description,
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[600],
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            SizedBox(height: 12),
            AppButton(
              text: '使用此算法',
              onPressed: () => widget.onAlgorithmSelected?.call(algorithm),
              size: AppButtonSize.small,
              fullWidth: true,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchBar() {
    return Container(
      margin: EdgeInsets.all(16),
      child: AppTextField(
        hint: '搜索算法名称、类型或功能...',
        prefixIcon: Icons.search,
        onChanged: (value) {
          setState(() {
            _searchQuery = value;
            _applyFilter();
          });
        },
      ),
    );
  }

  Widget _buildFilterBar() {
    return Container(
      margin: EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: [
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  FilterChip(
                    label: Text(_filter.type ?? '全部类型'),
                    onSelected: (_) => _showTypeFilter(),
                    selected: _filter.type != null,
                  ),
                  SizedBox(width: 8),
                  FilterChip(
                    label: Text(_filter.speed ?? '全部速度'),
                    onSelected: (_) => _showSpeedFilter(),
                    selected: _filter.speed != null,
                  ),
                  SizedBox(width: 8),
                  FilterChip(
                    label: Text(_filter.quality ?? '全部质量'),
                    onSelected: (_) => _showQualityFilter(),
                    selected: _filter.quality != null,
                  ),
                ],
              ),
            ),
          ),
          IconButton(
            icon: Icon(Icons.clear_all),
            onPressed: _clearFilter,
          ),
        ],
      ),
    );
  }

  Widget _buildAlgorithmCard(Algorithm algorithm) {
    final isSelected = widget.selectedAlgorithm?.id == algorithm.id;

    return Container(
      margin: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: AppCard(
        onTap: () => widget.onAlgorithmSelected?.call(algorithm),
        type: isSelected ? AppCardType.elevated : AppCardType.basic,
        child: Container(
          decoration: isSelected
              ? BoxDecoration(
                  border: Border.all(
                    color: Theme.of(context).primaryColor,
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(16),
                )
              : null,
          child: ListTile(
            contentPadding: EdgeInsets.all(16),
            leading: CircleAvatar(
              backgroundColor: Colors.blue.withOpacity(0.1),
              child: Icon(Icons.psychology, color: Colors.blue),
            ),
            title: Row(
              children: [
                Expanded(
                  child: Text(
                    algorithm.name,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                IconButton(
                  icon: Icon(
                    _favoriteAlgorithms.contains(algorithm.id)
                        ? Icons.favorite
                        : Icons.favorite_border,
                    ),
                  onPressed: () => _toggleFavorite(algorithm.id),
                ),
              ],
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(height: 4),
                Row(
                  children: [
                    Chip(
                      label: Text(algorithm.type.displayName),
                      backgroundColor: Colors.blue.withOpacity(0.1),
                      labelStyle: TextStyle(fontSize: 10),
                    ),
                    SizedBox(width: 8),
                    Chip(
                      label: Text(algorithm.speed.displayName),
                      backgroundColor: Colors.green.withOpacity(0.1),
                      labelStyle: TextStyle(fontSize: 10),
                    ),
                    SizedBox(width: 8),
                    Chip(
                      label: Text(algorithm.quality.displayName),
                      backgroundColor: Colors.orange.withOpacity(0.1),
                      labelStyle: TextStyle(fontSize: 10),
                    ),
                  ],
                ),
                SizedBox(height: 8),
                Row(
                  children: [
                    _buildRatingStars(algorithm.rating),
                    SizedBox(width: 8),
                    Text(
                      algorithm.rating.toStringAsFixed(1),
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                    Spacer(),
                    Text(
                      '${algorithm.processingTime}秒',
                      style: TextStyle(
                        color: Colors.grey[600],
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            trailing: isSelected
                ? Icon(Icons.check_circle, color: Colors.green)
                : Icon(Icons.arrow_forward_ios),
          ),
        ),
      ),
    );
  }

  Widget _buildRatingStars(double rating) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(5, (index) {
        return Icon(
          index < rating.round() ? Icons.star : Icons.star_border,
          size: 16,
          color: Colors.orange,
        );
      }),
    );
  }

  void _loadAlgorithms() async {
    // 从API加载算法列表
    try {
      final algorithms = await AlgorithmService.getAlgorithms();
      setState(() {
        _algorithms = algorithms;
        _filteredAlgorithms = algorithms;
      });
    } catch (e) {
      _showError('加载算法列表失败: $e');
    }
  }

  void _loadRecommendedAlgorithms() async {
    if (widget.imageFile == null) return;

    try {
      final recommended = await AlgorithmService.getRecommendedAlgorithms(
        widget.imageFile!,
      );
      setState(() {
        _recommendedAlgorithms = recommended;
      });
    } catch (e) {
      _showError('加载推荐算法失败: $e');
    }
  }

  void _applyFilter() {
    setState(() {
      _filteredAlgorithms = _algorithms.where((algorithm) {
        // 搜索过滤
        if (_searchQuery.isNotEmpty) {
          final query = _searchQuery.toLowerCase();
          if (!algorithm.name.toLowerCase().contains(query) &&
              !algorithm.description.toLowerCase().contains(query)) {
            return false;
          }
        }

        // 类型过滤
        if (_filter.type != null &&
            algorithm.type.displayName != _filter.type) {
          return false;
        }

        // 速度过滤
        if (_filter.speed != null &&
            algorithm.speed.displayName != _filter.speed) {
          return false;
        }

        // 质量过滤
        if (_filter.quality != null &&
            algorithm.quality.displayName != _filter.quality) {
          return false;
        }

        return true;
      }).toList();
    });
  }

  void _toggleFavorite(String algorithmId) async {
    setState(() {
      if (_favoriteAlgorithms.contains(algorithmId)) {
        _favoriteAlgorithms.remove(algorithmId);
      } else {
        _favoriteAlgorithms.add(algorithmId);
      }
    });

    // 保存到本地存储
    await StorageService.saveFavoriteAlgorithms(_favoriteAlgorithms);
  }

  void _showTypeFilter() {
    showModalBottomSheet(
      context: context,
      builder: (context) => _buildFilterBottomSheet(
        '选择算法类型',
        AlgorithmType.values.map((type) => type.displayName).toList(),
        _filter.type,
        (selected) {
          setState(() {
            _filter.type = selected;
            _applyFilter();
          });
        },
      ),
    );
  }

  void _showSpeedFilter() {
    showModalBottomSheet(
      context: context,
      builder: (context) => _buildFilterBottomSheet(
        '选择处理速度',
        ProcessingSpeed.values.map((speed) => speed.displayName).toList(),
        _filter.speed,
        (selected) {
          setState(() {
            _filter.speed = selected;
            _applyFilter();
          });
        },
      ),
    );
  }

  void _showQualityFilter() {
    showModalBottomSheet(
      context: context,
      builder: (context) => _buildFilterBottomSheet(
        '选择效果质量',
        QualityLevel.values.map((quality) => quality.displayName).toList(),
        _filter.quality,
        (selected) {
          setState(() {
            _filter.quality = selected;
            _applyFilter();
          });
        },
      ),
    );
  }

  Widget _buildFilterBottomSheet(
    String title,
    List<String> options,
    String? selected,
    Function(String?) onSelected,
  ) {
    return Container(
      padding: EdgeInsets.all(16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 24),
          ...options.map((option) {
            return ListTile(
              title: Text(option),
              trailing: selected == option ? Icon(Icons.check) : null,
              onTap: () {
                Navigator.pop(context);
                onSelected(selected == option ? null : option);
              },
            );
          }).toList(),
        ],
      ),
    );
  }

  void _clearFilter() {
    setState(() {
      _filter = AlgorithmFilter();
      _applyFilter();
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }
}

class AlgorithmFilter {
  String? type;
  String? speed;
  String? quality;
}
```

---

## 📱 响应式组件设计

### 响应式布局容器 (ResponsiveContainer)

#### 组件实现
```dart
class ResponsiveContainer extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final double? maxWidth;
  final bool centerContent;

  const ResponsiveContainer({
    Key? key,
    required this.child,
    this.padding,
    this.maxWidth,
    this.centerContent = true,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final effectiveMaxWidth = maxWidth ?? _getMaxWidthForScreen(screenWidth);

    return Container(
      width: double.infinity,
      padding: padding,
      child: centerContent
          ? Center(
              child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: effectiveMaxWidth),
                child: child,
              ),
            )
          : ConstrainedBox(
              constraints: BoxConstraints(maxWidth: effectiveMaxWidth),
              child: child,
            ),
    );
  }

  double _getMaxWidthForScreen(double screenWidth) {
    if (screenWidth < 768) {
      return screenWidth; // Mobile: full width
    } else if (screenWidth < 1024) {
      return 768; // Tablet: limit width
    } else if (screenWidth < 1440) {
      return 1024; // Desktop: standard width
    } else {
      return 1200; // Large desktop: wider but not too wide
    }
  }
}
```

### 响应式网格 (ResponsiveGrid)

#### 组件实现
```dart
class ResponsiveGrid extends StatelessWidget {
  final List<Widget> children;
  final double spacing;
  final double runSpacing;
  final EdgeInsetsGeometry? padding;
  final int? mobileColumns;
  final int? tabletColumns;
  final int? desktopColumns;

  const ResponsiveGrid({
    Key? key,
    required this.children,
    this.spacing = 16,
    this.runSpacing = 16,
    this.padding,
    this.mobileColumns = 1,
    this.tabletColumns = 2,
    this.desktopColumns = 3,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = _getColumnsForWidth(constraints.maxWidth);

        return Container(
          padding: padding,
          child: GridView.builder(
            shrinkWrap: true,
            physics: NeverScrollableScrollPhysics(),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: columns,
              crossAxisSpacing: spacing,
              mainAxisSpacing: runSpacing,
              childAspectRatio: _getChildAspectRatio(columns),
            ),
            itemCount: children.length,
            itemBuilder: (context, index) => children[index],
          ),
        );
      },
    );
  }

  int _getColumnsForWidth(double width) {
    if (width < 768) {
      return mobileColumns!;
    } else if (width < 1024) {
      return tabletColumns!;
    } else {
      return desktopColumns!;
    }
  }

  double _getChildAspectRatio(int columns) {
    switch (columns) {
      case 1:
        return 16 / 9; // Mobile: wider
      case 2:
        return 1; // Tablet: square
      case 3:
        return 4 / 3; // Desktop: slightly taller
      default:
        return 1;
    }
  }
}
```

---

## 🎯 交互优化策略

### 手势操作支持

#### 图片查看手势
```dart
class InteractiveImageView extends StatefulWidget {
  final ImageProvider image;
  final double? initialScale;
  final bool allowZoom;
  final bool allowPan;

  const InteractiveImageView({
    Key? key,
    required this.image,
    this.initialScale,
    this.allowZoom = true,
    this.allowPan = true,
  }) : super(key: key);

  @override
  State<InteractiveImageView> createState() => _InteractiveImageViewState();
}

class _InteractiveImageViewState extends State<InteractiveImageView>
    with SingleTickerProviderStateMixin {
  TransformationController? _transformationController;
  late AnimationController _animationController;
  Animation<Matrix4>? _animation;

  @override
  void initState() {
    super.initState();
    _transformationController = TransformationController();
    _animationController = AnimationController(
      duration: Duration(milliseconds: 300),
      vsync: this,
    );
  }

  @override
  void dispose() {
    _transformationController?.dispose();
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return InteractiveViewer(
      transformationController: _transformationController,
      onInteractionEnd: (details) {
        if (details.velocity.pixelsPerSecond.dx.abs() > 600 ||
            details.velocity.pixelsPerSecond.dy.abs() > 600) {
          _resetTransform();
        }
      },
      child: Image(
        image: widget.image,
        fit: BoxFit.contain,
      ),
    );
  }

  void _resetTransform() {
    _animation = Matrix4Tween(
      begin: _transformationController!.value,
      end: Matrix4.identity(),
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));

    _animationController.reset();
    _animationController.forward();
  }
}
```

### 动画效果增强

#### 页面转场动画
```dart
class CustomPageRoute<T> extends PageRouteBuilder<T> {
  final Widget child;
  final Duration duration;
  final Offset? beginOffset;

  CustomPageRoute({
    required this.child,
    this.duration = const Duration(milliseconds: 300),
    this.beginOffset,
  }) : super(
          pageBuilder: (context, animation, secondaryAnimation) => child,
          transitionDuration: duration,
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            final offset = beginOffset ?? Offset(1.0, 0.0);
            final curve = Curves.easeInOut;

            var offsetAnimation = Tween<Offset>(
              begin: offset,
              end: Offset.zero,
            ).chain(CurveTween(curve: curve));

            var fadeAnimation = Tween<double>(
              begin: 0.0,
              end: 1.0,
            ).chain(CurveTween(curve: curve));

            return SlideTransition(
              position: animation.drive(offsetAnimation),
              child: FadeTransition(
                opacity: animation.drive(fadeAnimation),
                child: child,
              ),
            );
          },
        );
```

#### 按钮点击动画
```dart
class AnimatedButton extends StatefulWidget {
  final Widget child;
  final VoidCallback onPressed;
  final Duration duration;

  const AnimatedButton({
    Key? key,
    required this.child,
    required this.onPressed,
    this.duration = const Duration(milliseconds: 150),
  }) : super(key: key);

  @override
  State<AnimatedButton> createState() => _AnimatedButtonState();
}

class _AnimatedButtonState extends State<AnimatedButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _opacityAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.duration,
      vsync: this,
    );

    _scaleAnimation = Tween<double>(
      begin: 1.0,
      end: 0.95,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));

    _opacityAnimation = Tween<double>(
      begin: 1.0,
      end: 0.8,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) {
        _controller.reverse();
        widget.onPressed();
      },
      onTapCancel: () => _controller.reverse(),
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Opacity(
              opacity: _opacityAnimation.value,
              child: widget.child,
            ),
          );
        },
      ),
    );
  }
}
```

---

## 📊 性能优化策略

### 图片懒加载
```dart
class LazyImage extends StatefulWidget {
  final String imageUrl;
  final Widget? placeholder;
  final Widget? errorWidget;
  final double? width;
  final double? height;
  final BoxFit fit;

  const LazyImage({
    Key? key,
    required this.imageUrl,
    this.placeholder,
    this.errorWidget,
    this.width,
    this.height,
    this.fit = BoxFit.cover,
  }) : super(key: key);

  @override
  State<LazyImage> createState() => _LazyImageState();
}

class _LazyImageState extends State<LazyImage> {
  bool _isVisible = false;
  ImageProvider? _imageProvider;

  @override
  void initState() {
    super.initState();
    // 预加载图片
    _preloadImage();
  }

  @override
  Widget build(BuildContext context) {
    return VisibilityDetector(
      key: Key('image_${widget.imageUrl}'),
      onVisibilityChanged: (visibilityInfo) {
        if (visibilityInfo.visibleFraction > 0.1 && !_isVisible) {
          setState(() {
            _isVisible = true;
          });
        }
      },
      child: _isVisible
          ? _buildImage()
          : widget.placeholder ?? _buildDefaultPlaceholder(),
    );
  }

  Widget _buildImage() {
    return Image(
      image: _imageProvider!,
      width: widget.width,
      height: widget.height,
      fit: widget.fit,
      errorBuilder: (context, error, stackTrace) {
        return widget.errorWidget ?? _buildDefaultErrorWidget();
      },
      loadingBuilder: (context, child, loadingProgress) {
        if (loadingProgress == null) return child;
        return widget.placeholder ?? _buildDefaultPlaceholder();
      },
    );
  }

  Widget _buildDefaultPlaceholder() {
    return Container(
      width: widget.width,
      height: widget.height,
      color: Colors.grey[200],
      child: Center(
        child: CircularProgressIndicator(),
      ),
    );
  }

  Widget _buildDefaultErrorWidget() {
    return Container(
      width: widget.width,
      height: widget.height,
      color: Colors.grey[200],
      child: Icon(Icons.error, color: Colors.grey[400]),
    );
  }

  void _preloadImage() {
    // 使用 precachedImage 预加载图片
    if (_imageProvider != null) {
      precacheImage(_imageProvider!, context);
    }
  }
}
```

### 列表性能优化
```dart
class OptimizedListView<T> extends StatelessWidget {
  final List<T> items;
  final Widget Function(BuildContext context, T item, int index) itemBuilder;
  final ScrollController? controller;
  final bool shrinkWrap;
  final EdgeInsetsGeometry? padding;

  const OptimizedListView({
    Key? key,
    required this.items,
    required this.itemBuilder,
    this.controller,
    this.shrinkWrap = false,
    this.padding,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      controller: controller,
      shrinkWrap: shrinkWrap,
      padding: padding,
      itemCount: items.length,
      // 使用 automaticKeepAlive 保持item状态
      addAutomaticKeepAlives: true,
      // 使用 cacheExtent 预渲染附近的项目
      cacheExtent: 250,
      itemBuilder: (context, index) {
        final item = items[index];
        return _buildOptimizedItem(context, item, index);
      },
    );
  }

  Widget _buildOptimizedItem(BuildContext context, T item, int index) {
    // 使用 AutomaticKeepAliveClientMixin 保持状态
    return _OptimizedListItem<T>(
      item: item,
      index: index,
      builder: itemBuilder,
    );
  }
}

class _OptimizedListItem<T> extends StatefulWidget {
  final T item;
  final int index;
  final Widget Function(BuildContext context, T item, int index) builder;

  const _OptimizedListItem({
    Key? key,
    required this.item,
    required this.index,
    required this.builder,
  }) : super(key: key);

  @override
  State<_OptimizedListItem<T>> createState() => _OptimizedListItemState<T>();
}

class _OptimizedListItemState<T> extends State<_OptimizedListItem<T>>
    with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return widget.builder(context, widget.item, widget.index);
  }
}
```

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**参考文档**: [设计系统](../design/01-design-system.md)、[业务组件](../design/05-business-components.md)