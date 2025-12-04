import React from 'react';
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
} from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Card from '@/components/Card';
import Badge from '@/components/Badge';
import { DatasetImage } from '../../types/dataset';

interface ImageCardProps {
  image: DatasetImage;
  onPress: (image: DatasetImage) => void;
  imageWidth?: number;
}

const ImageCard: React.FC<ImageCardProps> = ({
  image,
  onPress,
  imageWidth = 150,
}) => {
  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'foggy':
        return '有雾';
      case 'clear':
        return '无雾';
      case 'annotated':
        return '标注';
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

  const imageHeight = imageWidth; // 正方形比例

  return (
    <TouchableOpacity
      onPress={() => onPress(image)}
      activeOpacity={0.8}
      style={[
        styles.container,
        { width: imageWidth },
      ]}
    >
      <Card padding={0} margin={0} borderRadius={8}>
        <View style={[styles.imageContainer, { width: imageWidth, height: imageHeight }]}>
          <ImageLoader
            source={{ uri: image.image_url }}
            style={styles.image}
            resizeMode="cover"
          />

          {/* Type Badge */}
          <View style={styles.badgeContainer}>
            <Badge
              text={getTypeLabel(image.image_type)}
              variant={getBadgeVariant(image.image_type)}
              size="small"
            />
          </View>
        </View>

        <View style={styles.imageInfo}>
          <Text style={styles.filename} numberOfLines={1}>
            {image.filename}
          </Text>
        </View>
      </Card>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 12,
  },
  imageContainer: {
    position: 'relative',
    overflow: 'hidden',
  },
  image: {
    backgroundColor: '#f3f4f6',
    width: '100%',
    height: '100%',
  },
  badgeContainer: {
    position: 'absolute',
    top: 8,
    right: 8,
  },
  imageInfo: {
    padding: 8,
  },
  filename: {
    fontSize: 12,
    color: '#4b5563',
    textAlign: 'center',
  },
});

export default ImageCard;