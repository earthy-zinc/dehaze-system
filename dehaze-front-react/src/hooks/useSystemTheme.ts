import { ThemeEnum } from "@/enums/ThemeEnum";
import { useEffect, useState } from "react";

export default function useSystemTheme() {
  const [systemTheme, setSystemTheme] = useState<ThemeEnum>(ThemeEnum.LIGHT);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemTheme(media.matches ? ThemeEnum.DARK : ThemeEnum.LIGHT);

    const handleThemeChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? ThemeEnum.DARK : ThemeEnum.LIGHT);
    };
    media.addEventListener("change", handleThemeChange);

    return () => {
      media.removeEventListener("change", handleThemeChange);
    };
  }, []);
  return systemTheme;
}
