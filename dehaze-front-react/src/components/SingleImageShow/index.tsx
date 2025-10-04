import React from "react";

const SingleImageShow: React.FC<{ src: string }> = ({ src }) => {
  return (
    <img
      style={{
        width: "100%",
        height: "100%",
        objectFit: "contain",
      }}
      src={src}
      alt="result"
    />
  );
};
export default SingleImageShow;
