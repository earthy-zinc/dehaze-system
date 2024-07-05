import { PayloadAction, createSlice } from "@reduxjs/toolkit";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";
import defaultSettings from "@/settings";

const initialState = {
  settingsVisiable: false,
  tagsView: defaultSettings.tagsView,
  sidebarLogo: defaultSettings.sidebarLogo,
  fixedHeader: defaultSettings.fixedHeader,
  layout: defaultSettings.layout,
  themeColor: defaultSettings.themeColor,
  theme: defaultSettings.theme,
  watermarkEnabled: defaultSettings.watermarkEnabled,
};

type settingsType = typeof initialState;
type SettingsValue = boolean | string;

const settingsPersistConfig = {
  key: "settings",
  storage,
  whitelist: [
    "tagsView",
    "sidebarLogo",
    "fixedHeader",
    "layout",
    "themeColor",
    "theme",
    "watermarkEnabled",
  ],
};

const settingsSlice = createSlice({
  name: "settings",
  initialState,
  reducers: {
    toggleSettingsVisiable: (state) => {
      state.settingsVisiable = !state.settingsVisiable;
    },
    toggleWatermark: (state) => {
      state.watermarkEnabled = !state.watermarkEnabled;
    },
    toggleSidebarLogo: (state) => {
      state.sidebarLogo = !state.sidebarLogo;
    },
    changeTheme: (state, action: PayloadAction<string>) => {
      state.theme = action.payload;
    },
    changeThemeColor: (state, action) => {
      state.themeColor = action.payload;
    },
    changeLayout: (state, action: PayloadAction<string>) => {
      state.layout = action.payload;
    },
    changeSetting: (
      state,
      action: PayloadAction<{
        key: keyof settingsType;
        value: settingsType[keyof settingsType];
      }>
    ) => {
      const { key, value } = action.payload;
      if (key in state) {
        // @ts-expect-error 暂时忽略类型检查
        state[key] = value as SettingsValue;
      }
    },
  },
});

export default persistReducer(settingsPersistConfig, settingsSlice.reducer);
