/**
 * 算法上下文
 *
 * 维护当前选中的算法与算法树缓存，跨「算法选择→去雾处理→效果对比」流程共享。
 */
import type { Algorithm } from '@/types/algorithm';
import React, {
  createContext,
  useContext,
  useReducer,
  type ReactNode,
} from 'react';

interface AlgorithmState {
  /** 当前选中的算法 */
  current: Algorithm | null;
  /** 算法树缓存 */
  tree: Algorithm[];
}

type AlgorithmAction =
  | { type: 'SET_CURRENT'; algorithm: Algorithm | null }
  | { type: 'SET_TREE'; tree: Algorithm[] };

const initialState: AlgorithmState = {
  current: null,
  tree: [],
};

function algorithmReducer(
  state: AlgorithmState,
  action: AlgorithmAction,
): AlgorithmState {
  switch (action.type) {
    case 'SET_CURRENT':
      return { ...state, current: action.algorithm };
    case 'SET_TREE':
      return { ...state, tree: action.tree };
    default:
      return state;
  }
}

interface AlgorithmContextValue {
  state: AlgorithmState;
  setCurrentAlgorithm: (algorithm: Algorithm | null) => void;
  setAlgorithmTree: (tree: Algorithm[]) => void;
}

const AlgorithmContext = createContext<AlgorithmContextValue | undefined>(
  undefined,
);

export function AlgorithmProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(algorithmReducer, initialState);

  const value: AlgorithmContextValue = {
    state,
    setCurrentAlgorithm: algorithm =>
      dispatch({ type: 'SET_CURRENT', algorithm }),
    setAlgorithmTree: tree => dispatch({ type: 'SET_TREE', tree }),
  };

  return (
    <AlgorithmContext.Provider value={value}>
      {children}
    </AlgorithmContext.Provider>
  );
}

export function useAlgorithm(): AlgorithmContextValue {
  const ctx = useContext(AlgorithmContext);
  if (!ctx) {
    throw new Error('useAlgorithm 必须在 AlgorithmProvider 内使用');
  }
  return ctx;
}
