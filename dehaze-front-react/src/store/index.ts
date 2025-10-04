import appReducer from "@/store/modules/appSlice";
import imageShowReducer from "@/store/modules/imageShowSlice";
import settingsReducer from "@/store/modules/settingsSlice";
import userReducer from "@/store/modules/userSlice";
import { configureStore } from "@reduxjs/toolkit";
import { persistStore } from "redux-persist";
import permissionReducer from "./modules/permissionSlice";

const store = configureStore({
  reducer: {
    app: appReducer,
    settings: settingsReducer,
    user: userReducer,
    permission: permissionReducer,
    imageShow: imageShowReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

// 导出整个store的类型
export type RootState = ReturnType<typeof store.getState>;
export type DisPatchType = typeof store.dispatch;

export const persistor = persistStore(store);

export default store;
