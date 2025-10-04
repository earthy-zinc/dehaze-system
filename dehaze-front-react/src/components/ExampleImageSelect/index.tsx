import React from "react";
import "./index.scss";

interface ExampleImageSelectProps {
  urls: string[];
  onExampleSelect: (url: string) => void;
  className?: string;
}

const ExampleImageSelect: React.FC<ExampleImageSelectProps> = ({
  urls = [],
  onExampleSelect,
  className,
}) => {
  const ExampleImages = urls.map((url, index) => (
    <img
      key={url}
      src={url}
      className={"example-img"}
      alt={`example-${index}`}
      onClick={() => onExampleSelect(url)}
    />
  ));

  return (
    <div className={className + " example-container"}>
      <div className={"flex-center mt-15"}>选一张图片试一下吧</div>
      <div id="example-image-container">{ExampleImages}</div>
      <div className="example-logo">
        <img
          src="/logo/logo.png"
          alt="logo"
          style={{ width: "auto", height: "100px" }}
        />
      </div>
    </div>
  );
};

export default ExampleImageSelect;
