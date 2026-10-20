import { useDispatch } from "react-redux";
import type { DisPatchType, RootState } from "./index";

export const useAppDispatch = () => useDispatch<DisPatchType>();
export type { RootState };
