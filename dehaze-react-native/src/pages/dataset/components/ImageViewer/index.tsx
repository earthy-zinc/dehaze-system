import React, { useState, useCallback } from 'react';
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
import type { DatasetImage, DatasetItem } from '../../types/dataset';
import {
  formatFileSize,
  getTypeLabel,
  getBadgeVariant,
  getHazeLevelLabel,
} from '../../utils/imageLabels';
import { colors } from '@/theme/colors';

interface ImageViewerProps {
  visible: boolean;
  onClose: () => void;
  /** 主图片 */
  image: DatasetImage | null;
  /** 配对数据项（包含清晰图+多张有雾图） */
  item?: DatasetItem | null;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const ImageViewer: React.FC<ImageViewerProps> = ({
  visible,
  onClose,
  image,
  item,
}) => {
  // 构建可切换的图片列表：优先用 item 中的配对图，否则只显示当前图
  const buildImageList = useCallback((): DatasetImage[] => {
    if (item) {
      const list: DatasetImage[] = [];
      if (item.clearImage) list.push(item.clearImage);
      if (item.hazyImages && item.hazyImages.length > 0) {
        list.push(...item.hazyImages);
      }
      if (list.length > 0) return list;
    }
    return image ? [image] : [];
  }, [item, image]);

  const images = buildImageList();
  const [currentIndex, setCurrentIndex] = useState(0);

  // 当 image 或 item 变化时重置索引
  React.useEffect(() => {
    setCurrentIndex(0);
  }, [image?.id, item?.id]);

  if (!image && !item) return null;

  const current = images[currentIndex] || image;
  if (!current) return null;

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
            <Icon name="times" size={20} color={colors.text.inverse} />
          </TouchableOpacity>
        </View>

        {/* Image */}
        <View style={styles.imageContainer}>
          <Image
            source={{ uri: current.url }}
            style={styles.image}
            resizeMode="contain"
          />
        </View>

        {/* Pair Tabs (仅当有多张配对图片时显示) */}
        {images.length > 1 && (
          <View style={styles.tabsContainer}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {images.map((img, idx) => (
                <TouchableOpacity
                  key={img.id}
                  style={[styles.tab, idx === currentIndex && styles.activeTab]}
                  onPress={() => setCurrentIndex(idx)}
                >
                  <Text
                    style={[
                      styles.tabText,
                      idx === currentIndex && styles.activeTabText,
                    ]}
                  >
                    {getTypeLabel(img.type)}
                    {img.hazeLevel ? `-${getHazeLevelLabel(img.hazeLevel)}` : ''}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        )}

        {/* Image Info */}
        <View style={styles.imageInfo}>
          <Text style={styles.filename}>
            {current.fileName || `图片 #${current.id}`}
          </Text>

          <View style={styles.badgesRow}>
            <Badge
              text={getTypeLabel(current.type)}
              variant={getBadgeVariant(current.type)}
              size="medium"
            />
            {current.hazeLevel && (
              <Badge
                text={getHazeLevelLabel(current.hazeLevel)}
                variant="warning"
                size="medium"
              />
            )}
          </View>

          <View style={styles.details}>
            <View style={styles.detailItem}>
              <Text style={styles.detailLabel}>分辨率</Text>
              <Text style={styles.detailValue}>
                {current.width && current.height
                  ? `${current.width} × ${current.height}`
                  : '-'}
              </Text>
            </View>

            <View style={styles.detailItem}>
              <Text style={styles.detailLabel}>大小</Text>
              <Text style={styles.detailValue}>
                {current.formattedSize || formatFileSize(current.sizeBytes)}
              </Text>
            </View>

            {current.format ? (
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>格式</Text>
                <Text style={styles.detailValue}>{current.format}</Text>
              </View>
            ) : null}

            {current.sceneType ? (
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>场景</Text>
                <Text style={styles.detailValue}>{current.sceneType}</Text>
              </View>
            ) : null}

            {current.description ? (
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>描述</Text>
                <Text style={styles.detailValue}>{current.description}</Text>
              </View>
            ) : null}

            {typeof current.usageCount === 'number' ? (
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>使用次数</Text>
                <Text style={styles.detailValue}>{current.usageCount}</Text>
              </View>
            ) : null}
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
  tabsContainer: {
    backgroundColor: colors.background.primary,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.background.tertiary,
  },
  tab: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    backgroundColor: colors.background.tertiary,
    marginRight: 8,
  },
  activeTab: {
    backgroundColor: colors.secondary,
  },
  tabText: {
    fontSize: 13,
    color: colors.text.secondary,
    fontWeight: '500',
  },
  activeTabText: {
    color: colors.text.inverse,
  },
  imageInfo: {
    backgroundColor: colors.background.primary,
    padding: 20,
  },
  filename: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: 16,
  },
  badgesRow: {
    flexDirection: 'row',
    gap: 8,
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
    borderBottomColor: colors.background.tertiary,
  },
  detailLabel: {
    fontSize: 14,
    color: colors.text.secondary,
    fontWeight: '500',
  },
  detailValue: {
    fontSize: 14,
    color: colors.text.primary,
    flex: 1,
    textAlign: 'right',
    marginLeft: 20,
  },
});

export default ImageViewer;
