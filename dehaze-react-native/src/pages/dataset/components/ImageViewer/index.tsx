import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  Dimensions,
  ScrollView,
} from 'react-native';
import Modal from '@/components/Modal';
import Badge from '@/components/Badge';
import Icon from '@/components/Icon';
import { DatasetImage } from '../../types/dataset';

interface ImageViewerProps {
  visible: boolean;
  onClose: () => void;
  image: DatasetImage | null;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const ImageViewer: React.FC<ImageViewerProps> = ({
  visible,
  onClose,
  image,
}) => {
  if (!image) return null;

  const formatFileSize = (bytes: number) => {
    if (!bytes) return '-';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'foggy':
        return '有雾图像';
      case 'clear':
        return '无雾图像';
      case 'annotated':
        return '标注图像';
      default:
        return type;
    }
  };

  const getBadgeVariant = (type: string) => {
    switch (type) {
      case 'foggy':
        return 'foggy';
      case 'clear':
        return 'clear';
      case 'annotated':
        return 'annotated';
      default:
        return 'secondary';
    }
  };

  return (
    <Modal
      visible={visible}
      onClose={onClose}
      showCloseButton={false}
      animationType="fade"
    >
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        maximumZoomScale={3.0}
        minimumZoomScale={1.0}
      >
        {/* Close Button */}
        <View style={styles.closeButtonContainer}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={onClose}
            activeOpacity={0.8}
          >
            <Icon name="settings" size={24} color="#ffffff" />
          </TouchableOpacity>
        </View>

        {/* Image */}
        <View style={styles.imageContainer}>
          <Image
            source={{ uri: image.image_url }}
            style={styles.image}
            resizeMode="contain"
          />
        </View>

        {/* Image Info */}
        <View style={styles.imageInfo}>
          <Text style={styles.filename}>{image.filename}</Text>

          <View style={styles.typeContainer}>
            <Badge
              text={getTypeLabel(image.image_type)}
              variant={getBadgeVariant(image.image_type)}
              size="medium"
            />
          </View>

          <View style={styles.details}>
            <View style={styles.detailItem}>
              <Text style={styles.detailLabel}>尺寸</Text>
              <Text style={styles.detailValue}>
                {image.width} × {image.height}
              </Text>
            </View>

            <View style={styles.detailItem}>
              <Text style={styles.detailLabel}>大小</Text>
              <Text style={styles.detailValue}>
                {formatFileSize(image.file_size)}
              </Text>
            </View>

            {image.tags && (
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>标签</Text>
                <Text style={styles.detailValue}>{image.tags}</Text>
              </View>
            )}

            {image.description && (
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>描述</Text>
                <Text style={styles.detailValue}>{image.description}</Text>
              </View>
            )}
          </View>
        </View>
      </ScrollView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
  },
  closeButtonContainer: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 10,
  },
  closeButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageContainer: {
    minHeight: screenHeight * 0.6,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  image: {
    width: screenWidth,
    height: screenHeight * 0.6,
  },
  imageInfo: {
    backgroundColor: '#ffffff',
    padding: 20,
  },
  filename: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  typeContainer: {
    marginBottom: 20,
  },
  details: {
    gap: 12,
  },
  detailItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  detailLabel: {
    fontSize: 14,
    color: '#6b7280',
    fontWeight: '500',
  },
  detailValue: {
    fontSize: 14,
    color: '#1f2937',
    flex: 1,
    textAlign: 'right',
    marginLeft: 20,
  },
});

export default ImageViewer;