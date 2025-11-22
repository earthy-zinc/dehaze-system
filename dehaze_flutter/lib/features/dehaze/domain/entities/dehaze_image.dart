class DehazeImage {

  const DehazeImage({
    required this.id,
    required this.originalImagePath,
    required this.createdAt, required this.status, required this.parameters, this.processedImagePath,
    this.processedAt,
    this.metadata,
  });
  final String id;
  final String originalImagePath;
  final String? processedImagePath;
  final DateTime createdAt;
  final DateTime? processedAt;
  final ProcessingStatus status;
  final DehazeParameters parameters;
  final ProcessingMetadata? metadata;

  DehazeImage copyWith({
    String? id,
    String? originalImagePath,
    String? processedImagePath,
    DateTime? createdAt,
    DateTime? processedAt,
    ProcessingStatus? status,
    DehazeParameters? parameters,
    ProcessingMetadata? metadata,
  }) => DehazeImage(
      id: id ?? this.id,
      originalImagePath: originalImagePath ?? this.originalImagePath,
      processedImagePath: processedImagePath ?? this.processedImagePath,
      createdAt: createdAt ?? this.createdAt,
      processedAt: processedAt ?? this.processedAt,
      status: status ?? this.status,
      parameters: parameters ?? this.parameters,
      metadata: metadata ?? this.metadata,
    );

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is DehazeImage &&
        other.id == id &&
        other.originalImagePath == originalImagePath &&
        other.processedImagePath == processedImagePath &&
        other.createdAt == createdAt &&
        other.processedAt == processedAt &&
        other.status == status &&
        other.parameters == parameters &&
        other.metadata == metadata;
  }

  @override
  int get hashCode => id.hashCode ^
        originalImagePath.hashCode ^
        processedImagePath.hashCode ^
        createdAt.hashCode ^
        processedAt.hashCode ^
        status.hashCode ^
        parameters.hashCode ^
        metadata.hashCode;
}

enum ProcessingStatus { pending, processing, completed, failed, cancelled }

class DehazeParameters {

  const DehazeParameters({
    this.strength = 0.8,
    this.contrast = 1.2,
    this.brightness = 1.1,
    this.algorithm = DehazeAlgorithm.darkChannel,
    this.customParams = const {},
  });
  final double strength;
  final double contrast;
  final double brightness;
  final DehazeAlgorithm algorithm;
  final Map<String, dynamic> customParams;

  DehazeParameters copyWith({
    double? strength,
    double? contrast,
    double? brightness,
    DehazeAlgorithm? algorithm,
    Map<String, dynamic>? customParams,
  }) => DehazeParameters(
      strength: strength ?? this.strength,
      contrast: contrast ?? this.contrast,
      brightness: brightness ?? this.brightness,
      algorithm: algorithm ?? this.algorithm,
      customParams: customParams ?? this.customParams,
    );

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is DehazeParameters &&
        other.strength == strength &&
        other.contrast == contrast &&
        other.brightness == brightness &&
        other.algorithm == algorithm &&
        other.customParams == customParams;
  }

  @override
  int get hashCode => strength.hashCode ^
        contrast.hashCode ^
        brightness.hashCode ^
        algorithm.hashCode ^
        customParams.hashCode;
}

enum DehazeAlgorithm {
  darkChannel,
  atmosphericLight,
  retinex,
  colorAttenuation,
  custom,
}

class ProcessingMetadata {

  const ProcessingMetadata({
    required this.processingTime,
    required this.originalSize,
    required this.processedSize,
    required this.compressionRatio,
    required this.algorithmVersion,
    this.debugInfo = const {},
  });
  final Duration processingTime;
  final int originalSize;
  final int processedSize;
  final double compressionRatio;
  final String algorithmVersion;
  final Map<String, dynamic> debugInfo;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is ProcessingMetadata &&
        other.processingTime == processingTime &&
        other.originalSize == originalSize &&
        other.processedSize == processedSize &&
        other.compressionRatio == compressionRatio &&
        other.algorithmVersion == algorithmVersion &&
        other.debugInfo == debugInfo;
  }

  @override
  int get hashCode => processingTime.hashCode ^
        originalSize.hashCode ^
        processedSize.hashCode ^
        compressionRatio.hashCode ^
        algorithmVersion.hashCode ^
        debugInfo.hashCode;
}
