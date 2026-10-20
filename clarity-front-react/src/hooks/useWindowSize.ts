import { useSize } from "ahooks";

export const useWindowSize = () => {
  const size = useSize(document.body);
  if (size !== undefined) {
    const { width, height } = size;
    return { width, height };
  } else {
    return { width: window.innerWidth, height: window.innerHeight };
  }
};
