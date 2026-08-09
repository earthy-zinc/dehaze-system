import React, { useState } from 'react';
import {
  Image,
  View,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  ImageStyle,
} from 'react-native';
import { colors } from '@/theme/colors';

interface ImageLoaderProps {
  source: { uri: string } | number;
  style?: ImageStyle;
  containerStyle?: ViewStyle;
  placeholder?: React.ReactNode;
  onLoad?: () => void;
  onError?: () => void;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'repeat' | 'center';
}

const ImageLoader: React.FC<ImageLoaderProps> = ({
  source,
  style,
  containerStyle,
  placeholder,
  onLoad,
  onError,
  resizeMode = 'cover',
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const handleLoad = () => {
    setLoading(false);
    setError(false);
    onLoad?.();
  };

  const handleError = () => {
    setLoading(false);
    setError(true);
    onError?.();
  };

  return (
    <View style={[styles.container, containerStyle]}>
      {loading && (
        <View style={[styles.loadingContainer, style]}>
          <ActivityIndicator size="small" color={colors.primary} />
        </View>
      )}

      {error && (
        <View style={[styles.errorContainer, style]}>
          {placeholder || <View style={styles.errorPlaceholder} />}
        </View>
      )}

      <Image
        source={source}
        style={[
          styles.image,
          style,
          loading && styles.hidden,
          error && styles.hidden,
        ]}
        onLoad={handleLoad}
        onError={handleError}
        resizeMode={resizeMode}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  hidden: {
    opacity: 0,
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: colors.background.tertiary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: colors.background.tertiary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorPlaceholder: {
    width: 50,
    height: 50,
    backgroundColor: colors.border.light,
    borderRadius: 8,
  },
});

export default ImageLoader;