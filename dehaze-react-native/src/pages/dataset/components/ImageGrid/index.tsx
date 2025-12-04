import React, { useCallback } from 'react';
import {
  FlatList,
  StyleSheet,
  Dimensions,
} from 'react-native';
import ImageCard from '../ImageCard';
import { DatasetImage } from '../../types/dataset';

interface ImageGridProps {
  images: DatasetImage[];
  onImagePress: (image: DatasetImage) => void;
  onEndReached?: () => void;
  onRefresh?: () => void;
  refreshing?: boolean;
  isLoading?: boolean;
  numColumns?: number;
}

const { width: screenWidth } = Dimensions.get('window');

const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  onImagePress,
  onEndReached,
  onRefresh,
  refreshing = false,
  numColumns = 2,
}) => {
  const spacing = 12;
  const containerPadding = 20;
  const availableWidth = screenWidth - containerPadding * 2 - spacing * (numColumns - 1);
  const imageWidth = Math.floor(availableWidth / numColumns);

  const renderItem = useCallback(({ item }: { item: DatasetImage }) => (
    <ImageCard
      image={item}
      onPress={onImagePress}
      imageWidth={imageWidth}
    />
  ), [onImagePress, imageWidth]);

  const keyExtractor = useCallback((item: DatasetImage) => item.id.toString(), []);

  const getItemLayout = useCallback((data: any, index: number) => ({
    length: imageWidth + spacing,
    offset: (imageWidth + spacing) * Math.floor(index / numColumns),
    index,
  }), [imageWidth, spacing, numColumns]);

  const renderEmpty = useCallback(() => null, []);

  return (
    <FlatList
      data={images}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      numColumns={numColumns}
      contentContainerStyle={[
        styles.container,
        { paddingHorizontal: containerPadding },
      ]}
      columnWrapperStyle={numColumns > 1 ? styles.row : undefined}
      onEndReached={onEndReached}
      onEndReachedThreshold={0.5}
      onRefresh={onRefresh}
      refreshing={refreshing}
      ListEmptyComponent={renderEmpty}
      showsVerticalScrollIndicator={false}
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      initialNumToRender={10}
      windowSize={10}
      getItemLayout={getItemLayout}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    paddingBottom: 20,
  },
  row: {
    justifyContent: 'space-between',
  },
});

export default ImageGrid;