import { configureStore } from "@reduxjs/toolkit";
import { persistStore } from "redux-persist";
import appReducer from "@/store/modules/appSlice";
import settingsReducer from "@/store/modules/settingsSlice";

const store = configureStore({
  reducer: {
    app: appReducer,
    settings: settingsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

// 导出整个store的类型
export type RootState = ReturnType<typeof store.getState>;

export const persistor = persistStore(store);

export default store;
