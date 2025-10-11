import { ThemeEnum } from "@/enums/ThemeEnum";
import { useEffect, useState } from "react";

export default function useSystemTheme() {
  const [systemTheme, setSystemTheme] = useState<ThemeEnum>(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    return media.matches ? ThemeEnum.DARK : ThemeEnum.LIGHT;
  });

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const handleThemeChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? ThemeEnum.DARK : ThemeEnum.LIGHT);
    };

    // 添加监听器
    media.addEventListener("change", handleThemeChange);

    // 清理监听器
    return () => {
      media.removeEventListener("change", handleThemeChange);
    };
  }, []);

  return systemTheme;
}
