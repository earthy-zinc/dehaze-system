/**
 * 章节锚点滚动 hook
 *
 * 封装章节锚点条相关的滚动测量与激活同步逻辑：
 *  - 记录每个章节的布局位置
 *  - 点击锚点滚动定位到对应章节
 *  - 滚动过程中同步当前激活章节
 * 与页面数据加载、渲染解耦，页面仅需把返回值挂到对应组件上。
 */
import { useCallback, useRef, useState } from 'react';
import {
  ScrollView,
  type LayoutChangeEvent,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from 'react-native';

export interface SectionDef {
  key: string;
  label: string;
  icon: string;
}

interface SectionScrollResult {
  scrollRef: React.RefObject<ScrollView | null>;
  activeSection: string;
  handleSectionPress: (key: string) => void;
  handleSectionLayout: (key: string) => (e: LayoutChangeEvent) => void;
  handleScroll: (event: NativeSyntheticEvent<NativeScrollEvent>) => void;
}

/** 锚点点击后滚动目标位置相对章节顶部的偏移 */
const SCROLL_OFFSET = 140;
/** 滚动激活判断时相对视口顶部的偏移 */
const ACTIVATE_OFFSET = 160;

export function useSectionScroll(sections: readonly SectionDef[]): SectionScrollResult {
  const scrollRef = useRef<ScrollView>(null);
  const sectionLayouts = useRef<Record<string, number>>({});
  const [activeSection, setActiveSection] = useState(sections[0]?.key ?? '');

  /** 章节锚点点击 → 滚动到对应位置 */
  const handleSectionPress = useCallback((key: string) => {
    setActiveSection(key);
    const y = sectionLayouts.current[key];
    if (y !== undefined && scrollRef.current) {
      scrollRef.current.scrollTo({ y: y - SCROLL_OFFSET, animated: true });
    }
  }, []);

  /** 测量章节位置 */
  const handleSectionLayout = useCallback(
    (key: string) => (e: LayoutChangeEvent) => {
      sectionLayouts.current[key] = e.nativeEvent.layout.y;
    },
    [],
  );

  /** 滚动时更新激活章节 */
  const handleScroll = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      const y = event.nativeEvent.contentOffset.y + ACTIVATE_OFFSET;
      let current = activeSection;
      for (const section of sections) {
        const top = sectionLayouts.current[section.key];
        if (top !== undefined && top <= y) {
          current = section.key;
        }
      }
      if (current !== activeSection) {
        setActiveSection(current);
      }
    },
    [activeSection, sections],
  );

  return {
    scrollRef,
    activeSection,
    handleSectionPress,
    handleSectionLayout,
    handleScroll,
  };
}