import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../domain/entities/dehaze_image.dart';

class DehazeControlsWidget extends StatefulWidget {

  const DehazeControlsWidget({
    required this.currentParameters, required this.availableAlgorithms, required this.isProcessing, required this.onParametersChanged, required this.onProcessImage, super.key,
  });
  final DehazeParameters currentParameters;
  final List<DehazeAlgorithm> availableAlgorithms;
  final bool isProcessing;
  final void Function(DehazeParameters) onParametersChanged;
  final void Function(String) onProcessImage;

  @override
  State<DehazeControlsWidget> createState() => _DehazeControlsWidgetState();
}

class _DehazeControlsWidgetState extends State<DehazeControlsWidget> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();
  final TextEditingController _imagePathController = TextEditingController();

  DehazeParameters _parameters = const DehazeParameters();

  @override
  void initState() {
    super.initState();
    _parameters = widget.currentParameters;
  }

  @override
  void dispose() {
    _imagePathController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('去雾参数设置', style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _imagePathController,
                  decoration: const InputDecoration(
                    labelText: '图片路径',
                    border: OutlineInputBorder(),
                    hintText: '请输入图片路径或选择图片',
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return '请输入图片路径';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 6),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: widget.isProcessing ? null : _selectImage,
                    icon: const Icon(Icons.photo_library, size: 18),
                    label: const Text('选择图片'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<DehazeAlgorithm>(
                  initialValue: _parameters.algorithm,
                  decoration: const InputDecoration(
                    labelText: '去雾算法',
                    border: OutlineInputBorder(),
                  ),
                  items: widget.availableAlgorithms.map((algorithm) => DropdownMenuItem(
                      value: algorithm,
                      child: Text(_getAlgorithmName(algorithm)),
                    )).toList(),
                  onChanged: widget.isProcessing
                      ? null
                      : (algorithm) {
                          if (algorithm != null) {
                            _updateParameters(
                              _parameters.copyWith(algorithm: algorithm),
                            );
                          }
                        },
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Text(
                      '强度: ${(_parameters.strength * 100).toInt()}%',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    const Spacer(),
                    Text(
                      '${(_parameters.strength * 100).toInt()}%',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Slider(
                  value: _parameters.strength,
                  divisions: 100,
                  onChanged: widget.isProcessing
                      ? null
                      : (value) {
                          _updateParameters(
                            _parameters.copyWith(strength: value),
                          );
                        },
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Text(
                      '对比度: ${_parameters.contrast.toStringAsFixed(1)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    const Spacer(),
                    Text(
                      _parameters.contrast.toStringAsFixed(1),
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Slider(
                  value: _parameters.contrast,
                  max: 3,
                  divisions: 30,
                  onChanged: widget.isProcessing
                      ? null
                      : (value) {
                          _updateParameters(
                            _parameters.copyWith(contrast: value),
                          );
                        },
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Text(
                      '亮度: ${_parameters.brightness.toStringAsFixed(1)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    const Spacer(),
                    Text(
                      _parameters.brightness.toStringAsFixed(1),
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Slider(
                  value: _parameters.brightness,
                  max: 3,
                  divisions: 30,
                  onChanged: widget.isProcessing
                      ? null
                      : (value) {
                          _updateParameters(
                            _parameters.copyWith(brightness: value),
                          );
                        },
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: widget.isProcessing ? null : _processImage,
                    icon: widget.isProcessing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.play_arrow),
                    label: Text(widget.isProcessing ? '处理中...' : '开始去雾'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      backgroundColor: widget.isProcessing
                          ? Colors.grey
                          : Theme.of(context).primaryColor,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );

  void _updateParameters(DehazeParameters newParameters) {
    setState(() {
      _parameters = newParameters;
    });
    widget.onParametersChanged(newParameters);
  }

  Future<void> _selectImage() async {
    final result = await showDialog<String>(
      context: context,
      builder: (context) => const ImagePathDialog(),
    );

    if (result != null) {
      _imagePathController.text = result;
    }
  }

  void _processImage() {
    if (_formKey.currentState?.validate() ?? false) {
      widget.onProcessImage(_imagePathController.text);
    }
  }

  String _getAlgorithmName(DehazeAlgorithm algorithm) {
    switch (algorithm) {
      case DehazeAlgorithm.darkChannel:
        return '暗通道先验';
      case DehazeAlgorithm.atmosphericLight:
        return '大气光估计';
      case DehazeAlgorithm.retinex:
        return 'Retinex理论';
      case DehazeAlgorithm.colorAttenuation:
        return '颜色衰减先验';
      case DehazeAlgorithm.custom:
        return '自定义算法';
    }
  }
}

class ImagePathDialog extends StatelessWidget {
  const ImagePathDialog({super.key});

  @override
  Widget build(BuildContext context) => AlertDialog(
      title: const Text('选择图片'),
      content: const Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('请选择图片获取方式：'),
          SizedBox(height: 16),
          Text('• 相机\n• 相册\n• 文件路径'),
        ],
      ),
      actions: [
        TextButton(onPressed: () => context.pop(), child: const Text('取消')),
        ElevatedButton(
          onPressed: () {
            // 这里可以通过状态管理或其他方式传递选择结果
            context.pop();
          },
          child: const Text('选择示例图片'),
        ),
      ],
    );
}
