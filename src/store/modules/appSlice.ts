import defaultSettings from "@/settings";

import { DeviceEnum } from "@/enums/DeviceEnum";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { PayloadAction, createSlice } from "@reduxjs/toolkit";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

const initialState = {
  device: DeviceEnum.DESKTOP,
  size: defaultSettings.size,
  language: defaultSettings.language,
  sidebarStatus: SidebarStatusEnum.CLOSED,
  activeTopMenuPath: "",
};

export const appSlice = createSlice({
  name: "app",
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebarStatus =
        state.sidebarStatus === SidebarStatusEnum.CLOSED
          ? SidebarStatusEnum.OPENED
          : SidebarStatusEnum.CLOSED;
    },
    closeSideBar: (state) => {
      state.sidebarStatus = SidebarStatusEnum.CLOSED;
    },
    openSideBar: (state) => {
      state.sidebarStatus = SidebarStatusEnum.OPENED;
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
