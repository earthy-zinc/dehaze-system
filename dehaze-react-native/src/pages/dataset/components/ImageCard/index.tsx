import React, { useRef } from 'react';
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
  Animated,
} from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Card from '@/components/Card';
import Badge from '@/components/Badge';
import type { DatasetImage } from '../../types/dataset';
import {
  getTypeLabel,
  getBadgeVariant,
  getHazeLevelLabel,
} from '../../utils/imageLabels';

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
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.95,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const imageHeight = imageWidth;
  const displayUrl = image.thumbnailUrl || image.url;
  const hazeLabel = getHazeLevelLabel(image.hazeLevel);

  return (
    <TouchableOpacity
      onPress={() => onPress(image)}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
      style={[styles.container, { width: imageWidth }]}
    >
      <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
        <Card padding={0} margin={0} borderRadius={12}>
          <View
            style={[styles.imageContainer, { width: imageWidth, height: imageHeight }]}
          >
            <ImageLoader
              source={{ uri: displayUrl }}
              style={styles.image}
              resizeMode="cover"
            />

            <View style={styles.badgeContainer}>
              <Badge
                text={getTypeLabel(image.type)}
                variant={getBadgeVariant(image.type)}
                size="small"
              />
            </View>

            {hazeLabel ? (
              <View style={styles.hazeBadgeContainer}>
                <Badge text={hazeLabel} variant="warning" size="small" />
              </View>
            ) : null}
          </View>

          <View style={styles.imageInfo}>
            <Text style={styles.filename} numberOfLines={1}>
              {image.fileName || `#${image.id}`}
            </Text>
          </View>
        </Card>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 0,
  },
  imageContainer: {
    position: 'relative',
    overflow: 'hidden',
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
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
  hazeBadgeContainer: {
    position: 'absolute',
    top: 8,
    left: 8,
  },
  imageInfo: {
    padding: 10,
  },
  filename: {
    fontSize: 12,
    color: '#4b5563',
    textAlign: 'center',
  },
});

export default ImageCard;
