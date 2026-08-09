import { create } from "zustand";
import type { Algorithm, PredictionResultVO } from "dehaze-sdk-js";

export interface ProcessImage {
  url: string;
  name: string;
  width?: number;
  height?: number;
  size?: number;
  cleanUrl?: string;
}

interface ProcessState {
  image: ProcessImage | null;
  algorithm: Algorithm | null;
  result: PredictionResultVO | null;
  setImage: (image: ProcessImage) => void;
  setAlgorithm: (algorithm: Algorithm) => void;
  setResult: (result: PredictionResultVO) => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessState>()((set) => ({
  image: null,
  algorithm: null,
  result: null,
  setImage: (image) => set({ image }),
  setAlgorithm: (algorithm) => set({ algorithm }),
  setResult: (result) => set({ result }),
  reset: () => set({ image: null, algorithm: null, result: null }),
}));