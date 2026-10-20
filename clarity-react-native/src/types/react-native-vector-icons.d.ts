/**
 * react-native-vector-icons 本地类型声明
 *
 * react-native-vector-icons v10 未自带类型定义，
 * 而社区 @types/react-native-vector-icons 依赖旧版 @types/react-native，
 * 与 React Native 0.81 内置类型冲突。
 * 此处直接基于 RN 内置类型声明 Ionicons 子模块，避免引入冲突的 @types 包。
 */
declare module 'react-native-vector-icons/Ionicons' {
  import type { Component } from 'react';
  import type { TextProps, TextStyle } from 'react-native';

  export interface IoniconsProps extends TextProps {
    /** 图标名称 */
    name: string;
    /** 图标尺寸（px） */
    size?: number;
    /** 图标颜色 */
    color?: string;
    style?: TextStyle;
  }

  export default class Ionicons extends Component<IoniconsProps> {}
}
