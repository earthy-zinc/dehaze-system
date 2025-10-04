import "./index.scss";
import React from "react";

const Loading: React.FC = () => {
  return (
    <div>
      <div className="loading">
        {Array.from({ length: 5 }, (_, i) => (
          <span key={i}></span>
        ))}
      </div>
      <span style={{ marginLeft: "8px" }}>正在生成图片中，请耐心等候</span>
    </div>
  );
};

export default Loading;
