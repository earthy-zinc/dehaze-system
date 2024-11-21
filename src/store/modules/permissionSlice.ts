// permissionSlice.ts
import MenuAPI from "@/api/menu";
import { RouteVO } from "@/api/menu/model";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import React from "react";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

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
      tmpRoute.component = React.lazy(() => import("@/layout/index"));
    } else {
      tmpRoute.component = React.lazy(
        () => import(`../../pages/${tmpRoute.component}`)
      );
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

const permissionConfig = {
  key: "permission",
  storage,
};

export const { setRoutes, setMixLeftMenus } = permissionSlice.actions;
export default persistReducer(permissionConfig, permissionSlice.reducer);
