import { RootStackParamList } from '@/routes/navigator';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React from 'react';
import { StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import HeroSection from './components/HeroSection';
import ShowcaseSection from './components/ShowcaseSection';
import FeaturesSection from './components/FeaturesSection';
import AlgorithmSection from './components/AlgorithmSection';
import TechSpecsSection from './components/TechSpecsSection';
import FinalCTASection from './components/FinalCTASection';

type HomeScreenProps = NativeStackScreenProps<RootStackParamList, 'Home'>;

const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {

  // 导航处理函数
  const handleStartPress = () => {
    navigation.navigate('ImageInput');
  };

  const handleDatasetPress = () => {
    navigation.navigate('Dataset');
  };

  const handleImageInputPress = () => {
    navigation.navigate('ImageInput');
  };

  const handleAlgorithmSelectPress = () => {
    navigation.navigate('AlgorithmSelect');
  };

  const handleProcessingPress = () => {
    navigation.navigate('Processing');
  };

  const handleSideBySidePress = () => {
    navigation.navigate('SideBySide');
  };

  const handleOverlayPress = () => {
    navigation.navigate('Overlay');
  };

  const handleMagnifierPress = () => {
    navigation.navigate('Magnifier');
  };

  const handleFilterPress = () => {
    navigation.navigate('Filter');
  };

  const handleMetricsPress = () => {
    navigation.navigate('Metrics');
  };

  const handleDatasetManagePress = () => {
    navigation.navigate('Dataset');
  };

  const handleLearnMorePress = () => {
    navigation.navigate('Algorithm');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <HeroSection
          onStartPress={handleStartPress}
          onDatasetPress={handleDatasetPress}
        />

        <ShowcaseSection onPress={handleStartPress} />

        <FeaturesSection
          onImageInputPress={handleImageInputPress}
          onAlgorithmSelectPress={handleAlgorithmSelectPress}
          onProcessingPress={handleProcessingPress}
          onSideBySidePress={handleSideBySidePress}
          onOverlayPress={handleOverlayPress}
          onMagnifierPress={handleMagnifierPress}
          onFilterPress={handleFilterPress}
          onMetricsPress={handleMetricsPress}
          onDatasetManagePress={handleDatasetManagePress}
        />

        <AlgorithmSection onLearnMorePress={handleLearnMorePress} />

        <TechSpecsSection />

        <FinalCTASection onStartPress={handleStartPress} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
});

export default HomeScreen;
