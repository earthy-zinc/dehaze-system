import React from 'react'
import { View } from '@tarojs/components'

import PageLayout from '@/layout'
import HeroSection from './components/HeroSection'
import ShowcaseSection from './components/ShowcaseSection'
import WorkflowSection from './components/WorkflowSection'
import ToolsGrid from './components/ToolsGrid'
import AlgorithmSection from './components/AlgorithmSection'
import TechSpecs from './components/TechSpecs'
import CTASection from './components/CTASection'

import './index.less'

const HomePage: React.FC = () => {
  return (
    <PageLayout showTabbar currentRoute='/pages/home/index'>
      <View className='home-page'>
        <HeroSection />
        <ShowcaseSection />
        <WorkflowSection />
        <ToolsGrid />
        <AlgorithmSection />
        <TechSpecs />
        <CTASection />
      </View>
    </PageLayout>
  )
}

export default HomePage
