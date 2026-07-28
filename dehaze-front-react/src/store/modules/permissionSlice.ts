import { MenuAPI, RouteVO } from "dehaze-sdk-js";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

interface PermissionState {
  routes: RouteVO[];
  mixLeftMenus: RouteVO[];
}

const initialState: PermissionState = {
  routes: [],
  mixLeftMenus: [],
};

const hasPermission = (roles: string[], route: RouteVO): boolean => {
  if (!route.meta?.roles) return false;
  if (roles.includes("ROOT")) return true;
  return roles.some((role) => route.meta?.roles?.includes(role));
};

const filterAsyncRoutes = (routes: RouteVO[], roles: string[]): RouteVO[] => {
  const asyncRoutes: RouteVO[] = [];
  routes.forEach((route) => {
    const tmpRoute = { ...route } as RouteVO;
    if (!hasPermission(roles, tmpRoute)) return;
    if (tmpRoute.component === "Layout") {
      tmpRoute.component = undefined;
    }
    if (tmpRoute.children) {
      tmpRoute.children = filterAsyncRoutes(route.children, roles);
    }
    asyncRoutes.push(tmpRoute);
  });
  return asyncRoutes;
};

export const generateRoutes = createAsyncThunk(
  "permission/generateRoutes",
  async (roles: string[]) => {
    const response = await MenuAPI.getRoutes();
    return filterAsyncRoutes(response, roles);
  }
);

const permissionSlice = createSlice({
  name: "permission",
  initialState,
  reducers: {
    setRoutes: (state, action) => {
      state.routes = action.payload;
    },
    setMixLeftMenus: (state, action) => {
      const topMenuPath = action.payload;
      const matchedItem = state.routes.find(
        (item) => item.path === topMenuPath
      );
      if (matchedItem?.children) {
        state.mixLeftMenus = matchedItem.children;
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(generateRoutes.fulfilled, (state, action) => {
      state.routes = action.payload;
    });
  },
});

export const { setRoutes, setMixLeftMenus } = permissionSlice.actions;
export default permissionSlice.reducer;
