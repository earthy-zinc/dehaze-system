import { SizeEnum } from "./enums/SizeEnum";
import { LayoutEnum } from "./enums/LayoutEnum";
import { ThemeEnum } from "./enums/ThemeEnum";
import { LanguageEnum } from "./enums/LanguageEnum";

const { pkg } = __APP_INFO__;

const defaultSettings: AppSettings = {
  title: "图像去雾系统",
  version: pkg.version,
  showSettings: true,
  tagsView: false,
  fixedHeader: false,
  sidebarLogo: true,
  layout: LayoutEnum.TOP,
  theme: ThemeEnum.LIGHT,
  size: SizeEnum.DEFAULT,
  language: LanguageEnum.ZH_CN,
  themeColor: "#117B07",
  watermarkEnabled: false,
  watermarkContent: pkg.name,
};

export default defaultSettings;
