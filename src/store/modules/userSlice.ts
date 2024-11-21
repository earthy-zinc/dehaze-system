import AuthAPI from "@/api/auth";
import { LoginData } from "@/api/auth/model";
import UserAPI from "@/api/user";
import { UserInfo } from "@/api/user/model";
import { TOKEN_KEY } from "@/enums/CacheEnum";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

const initialState = {
  user: {
    roles: [] as string[],
    perms: [] as string[],
  } as UserInfo,
};

export const login = createAsyncThunk(
  "user/login",
  async (loginData: LoginData) => {
    const response = await AuthAPI.login(loginData);
    const { tokenType, accessToken } = response;
    localStorage.setItem(TOKEN_KEY, `${tokenType} ${accessToken}`);
    return response;
  }
);

export const getUserInfo = createAsyncThunk("user/getUserInfo", async () => {
  const response = await UserAPI.getInfo();
  if (!response || !response?.roles || response.roles.length <= 0) {
    throw new Error("Verification failed, please Login again.");
  }
  return response;
});

export const logout = createAsyncThunk("user/logout", async () => {
  await AuthAPI.logout();
  localStorage.removeItem(TOKEN_KEY);
  return {};
});

const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    resetToken: (state) => {
      localStorage.removeItem(TOKEN_KEY);
      state.user = { roles: [], perms: [] };
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getUserInfo.fulfilled, (state, action) => {
        state.user = action.payload;
      })
      .addCase(logout.fulfilled, (state) => {
        state.user = { roles: [], perms: [] };
      });
  },
});

const userPersistConfig = {
  key: "user",
  storage,
  whitelist: ["user"],
};

export const { resetToken } = userSlice.actions;
export default persistReducer(userPersistConfig, userSlice.reducer);
