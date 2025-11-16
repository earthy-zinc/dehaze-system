import '../../domain/entities/dehaze_image.dart';

class DehazeImageModel extends DehazeImage {
  const DehazeImageModel({
    required super.id,
    required super.originalImagePath,
    super.processedImagePath,
    required super.createdAt,
    super.processedAt,
    required super.status,
    required super.parameters,
    super.metadata,
  });

  factory DehazeImageModel.fromJson(Map<String, dynamic> json) {
    return DehazeImageModel(
      id: json['id'] as String,
      originalImagePath: json['originalImagePath'] as String,
      processedImagePath: json['processedImagePath'] as String?,
      createdAt: DateTime.parse(json['createdAt'] as String),
      processedAt: json['processedAt'] != null
          ? DateTime.parse(json['processedAt'] as String)
          : null,
      status: ProcessingStatus.values.firstWhere(
        (status) => status.toString() == 'ProcessingStatus.${json['status']}',
        orElse: () => ProcessingStatus.pending,
      ),
      parameters: DehazeParametersModel.fromJson(json['parameters'] as Map<String, dynamic>),
      metadata: json['metadata'] != null
          ? ProcessingMetadataModel.fromJson(json['metadata'] as Map<String, dynamic>)
          : null,
    );
  }

  factory DehazeImageModel.fromEntity(DehazeImage entity) {
    return DehazeImageModel(
      id: entity.id,
      originalImagePath: entity.originalImagePath,
      processedImagePath: entity.processedImagePath,
      createdAt: entity.createdAt,
      processedAt: entity.processedAt,
      status: entity.status,
      parameters: DehazeParametersModel.fromEntity(entity.parameters),
      metadata: entity.metadata != null
          ? ProcessingMetadataModel.fromEntity(entity.metadata!)
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'originalImagePath': originalImagePath,
      'processedImagePath': processedImagePath,
      'createdAt': createdAt.toIso8601String(),
      'processedAt': processedAt?.toIso8601String(),
      'status': status.toString().split('.').last,
      'parameters': DehazeParametersModel.fromEntity(parameters).toJson(),
      'metadata': metadata != null
          ? ProcessingMetadataModel.fromEntity(metadata!).toJson()
          : null,
    };
  }

  DehazeImage toEntity() {
    return DehazeImage(
      id: id,
      originalImagePath: originalImagePath,
      processedImagePath: processedImagePath,
      createdAt: createdAt,
      processedAt: processedAt,
      status: status,
      parameters: parameters,
      metadata: metadata,
    );
  }
}

class DehazeParametersModel extends DehazeParameters {
  const DehazeParametersModel({
    super.strength = 0.8,
    super.contrast = 1.2,
    super.brightness = 1.1,
    super.algorithm = DehazeAlgorithm.darkChannel,
    super.customParams = const {},
  });

  factory DehazeParametersModel.fromJson(Map<String, dynamic> json) {
    return DehazeParametersModel(
      strength: (json['strength'] as num?)?.toDouble() ?? 0.8,
      contrast: (json['contrast'] as num?)?.toDouble() ?? 1.2,
      brightness: (json['brightness'] as num?)?.toDouble() ?? 1.1,
      algorithm: DehazeAlgorithm.values.firstWhere(
        (algo) => algo.toString() == 'DehazeAlgorithm.${json['algorithm']}',
        orElse: () => DehazeAlgorithm.darkChannel,
      ),
      customParams: Map<String, dynamic>.from(json['customParams'] ?? {}),
    );
  }

  factory DehazeParametersModel.fromEntity(DehazeParameters entity) {
    return DehazeParametersModel(
      strength: entity.strength,
      contrast: entity.contrast,
      brightness: entity.brightness,
      algorithm: entity.algorithm,
      customParams: entity.customParams,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'strength': strength,
      'contrast': contrast,
      'brightness': brightness,
      'algorithm': algorithm.toString().split('.').last,
      'customParams': customParams,
    };
  }
}

class ProcessingMetadataModel extends ProcessingMetadata {
  const ProcessingMetadataModel({
    required super.processingTime,
    required super.originalSize,
    required super.processedSize,
    required super.compressionRatio,
    required super.algorithmVersion,
    super.debugInfo = const {},
  });

  factory ProcessingMetadataModel.fromJson(Map<String, dynamic> json) {
    return ProcessingMetadataModel(
      processingTime: Duration(milliseconds: json['processingTime'] as int),
      originalSize: json['originalSize'] as int,
      processedSize: json['processedSize'] as int,
      compressionRatio: (json['compressionRatio'] as num).toDouble(),
      algorithmVersion: json['algorithmVersion'] as String,
      debugInfo: Map<String, dynamic>.from(json['debugInfo'] ?? {}),
    );
  }

  factory ProcessingMetadataModel.fromEntity(ProcessingMetadata entity) {
    return ProcessingMetadataModel(
      processingTime: entity.processingTime,
      originalSize: entity.originalSize,
      processedSize: entity.processedSize,
      compressionRatio: entity.compressionRatio,
      algorithmVersion: entity.algorithmVersion,
      debugInfo: entity.debugInfo,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'processingTime': processingTime.inMilliseconds,
      'originalSize': originalSize,
      'processedSize': processedSize,
      'compressionRatio': compressionRatio,
      'algorithmVersion': algorithmVersion,
      'debugInfo': debugInfo,
    };
  }
}