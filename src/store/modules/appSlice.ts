import { PayloadAction, createSlice } from "@reduxjs/toolkit";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";
import { DeviceEnum } from "@/enums/DeviceEnum";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import defaultSettings from "@/settings";

const initialState = {
  device: DeviceEnum.DESKTOP,
  size: defaultSettings.size,
  language: defaultSettings.language,
  sidebarStatus: SidebarStatusEnum.OPENED,
  activeTopMenuPath: "",
};

export const appSlice = createSlice({
  name: "app",
  initialState,
  reducers: {
    toggleSidebar: (state, action: PayloadAction<SidebarStatusEnum>) => {
      state.sidebarStatus = action.payload;
    },
    toggleDevice: (state, action: PayloadAction<DeviceEnum>) => {
      state.device = action.payload;
    },
    changeSize: (state, action: PayloadAction<string>) => {
      state.size = action.payload;
    },
    changeLanguage: (state, action: PayloadAction<string>) => {
      state.language = action.payload;
    },
    activeTopMenu: (state, action: PayloadAction<string>) => {
      state.activeTopMenuPath = action.payload;
    },
  },
});

const appPersistConfig = {
  key: "app",
  storage,
  blackList: ["activeTopMenuPath"],
};

export default persistReducer(appPersistConfig, appSlice.reducer);
