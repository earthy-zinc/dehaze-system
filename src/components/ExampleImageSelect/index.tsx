import React from "react";
import "./index.scss";

interface ExampleImageSelectProps {
  urls: string[];
  onSelect: (url: string) => void;
}

const ExampleImageSelect: React.FC<ExampleImageSelectProps> = ({
  urls = [],
  onSelect,
}) => {
  const ExampleImages = urls.map((url, index) => (
    <img
      key={url}
      src={url}
      alt={`example-${index}`}
      onClick={() => onSelect(url)}
    />
  ));

  return (
    <div className="example-container">
      <div>选一张图片试一下吧</div>
      <div id="example-image-container">{ExampleImages}</div>
      <div className="example-logo">
        <img src="" alt="logo" />
      </div>
    </div>
  );
};

export default ExampleImageSelect;
