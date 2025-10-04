import React, { useEffect, useRef } from "react";
import Lazy from "./Lazy";
import "./index.scss";

interface LazyImgProps {
  url: string;
  renderer: () => void;
  title?: string;
  alt?: string;
}

const LazyImg: React.FC<LazyImgProps> = (props) => {
  const { url, renderer, title = "", alt = "" } = props;
  const lazyRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!lazyRef.current) return;
    const currentLazyRef = lazyRef.current;
    const lazy = new Lazy(true, {}, true);
    lazy.mount(currentLazyRef, url, () => {
      renderer();
    });
    return () => {
      lazy.unmount(currentLazyRef);
    };
  }, [renderer, url]);

  return (
    <div className="lazy__box">
      <div className="lazy__resource">
        <img ref={lazyRef} className="lazy__img" title={title} alt={alt} />
      </div>
    </div>
  );
};

export default LazyImg;
