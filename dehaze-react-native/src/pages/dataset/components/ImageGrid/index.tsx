import React, { useCallback } from 'react';
import {
  FlatList,
  StyleSheet,
  RefreshControl,
  View,
} from 'react-native';
import ImageCard from '../ImageCard';
import { DatasetImage } from '../../types/dataset';
import { useResponsive } from '@/hooks/useResponsive';
import { colors } from '@/theme/colors';

interface ImageGridProps {
  images: DatasetImage[];
  onImagePress: (image: DatasetImage) => void;
  onEndReached?: () => void;
  onRefresh?: () => void;
  refreshing?: boolean;
  isLoading?: boolean;
}

const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  onImagePress,
  onEndReached,
  onRefresh,
  refreshing = false,
}) => {
  const { width, isMobile, isTablet, containerPadding, spacing } = useResponsive();

  // 响应式列数
  const numColumns = isMobile ? 2 : isTablet ? 3 : 4;
  const availableWidth = width - containerPadding * 2 - spacing * (numColumns - 1);
  const imageWidth = Math.floor(availableWidth / numColumns);

  const renderItem = useCallback(({ item }: { item: DatasetImage }) => (
    <ImageCard
      image={item}
      onPress={onImagePress}
      imageWidth={imageWidth}
    />
  ), [onImagePress, imageWidth]);

  const keyExtractor = useCallback((item: DatasetImage) => item.id.toString(), []);

  const getItemLayout = useCallback(
    (_data: ArrayLike<DatasetImage> | null | undefined, index: number) => ({
      length: imageWidth + spacing,
      offset: (imageWidth + spacing) * Math.floor(index / numColumns),
      index,
    }),
    [imageWidth, spacing, numColumns],
  );

  const renderEmpty = useCallback(() => null, []);

  const ItemSeparator = useCallback(() => (
    <View style={{ height: spacing }} />
  ), [spacing]);

  return (
    <FlatList
      data={images}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      numColumns={numColumns}
      key={`grid-${numColumns}`} // 强制重新渲染当列数变化时
      contentContainerStyle={[
        styles.container,
        { paddingHorizontal: containerPadding },
      ]}
      columnWrapperStyle={numColumns > 1 ? [styles.row, { gap: spacing }] : undefined}
      onEndReached={onEndReached}
      onEndReachedThreshold={0.5}
      refreshControl={
        onRefresh ? (
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.secondary}
            colors={[colors.secondary]}
          />
        ) : undefined
      }
      ListEmptyComponent={renderEmpty}
      ItemSeparatorComponent={ItemSeparator}
      showsVerticalScrollIndicator={false}
      removeClippedSubviews={true}
      maxToRenderPerBatch={12}
      updateCellsBatchingPeriod={50}
      initialNumToRender={12}
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
    justifyContent: 'flex-start',
  },
});

export default ImageGrid;