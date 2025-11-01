export default defineAppConfig({
  pages: [
    "pages/login/index",
    "pages/dashboard/index",
    "pages/system/user/index",
    "pages/system/user/detail",
    "pages/system/role/index",
    "pages/system/role/detail",
    "pages/system/role/permission"
  ],
  window: {
    backgroundTextStyle: "light",
    navigationBarBackgroundColor: "#fff",
    navigationBarTitleText: "WeChat",
    navigationBarTextStyle: "black",
  },
});
