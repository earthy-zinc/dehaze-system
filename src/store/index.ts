import { configureStore } from "@reduxjs/toolkit";
import appReducer from "@/store/modules/appSlice";
import { persistStore } from "redux-persist";
import settingsReducer from "@/store/modules/settingsSlice";

const store = configureStore({
  reducer: {
    app: appReducer,
    settings: settingsReducer,
  },
});

// 导出整个store的类型
export type RootState = ReturnType<typeof store.getState>;

export const persistor = persistStore(store);

export default store;
