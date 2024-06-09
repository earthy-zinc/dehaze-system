import { useEffect, useState } from "react";

export default function useSystemTheme() {
  const [systemTheme, setSystemTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemTheme(media.matches ? "dark" : "light");

    const handleThemeChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? "dark" : "light");
    };
    media.addEventListener("change", handleThemeChange);

    return () => {
      media.removeEventListener("change", handleThemeChange);
    };
  }, []);
  return systemTheme;
}
