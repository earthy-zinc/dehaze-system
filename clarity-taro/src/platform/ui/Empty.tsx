// 小程序与 H5 直接用 Taroify 的 Empty（含 Image / Description 子组件）；RN 走 Empty.rn.tsx
export { Empty as default } from "@taroify/core";
export type {
  EmptyProps,
  EmptyImageProps,
  EmptyDescriptionProps,
} from "./empty-shared";
