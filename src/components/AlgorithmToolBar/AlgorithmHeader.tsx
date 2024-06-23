import React from "react";

interface AlgorithmHeaderProps {
  title: string;
  description: string;
}

const AlgorithmHeader: React.FC<AlgorithmHeaderProps> = ({
  title,
  description,
}) => {
  return (
    <>
      <h2 style={{ textAlign: "center", margin: "8px" }}>{title}</h2>
      <div style={{ margin: "8px" }}>{description}</div>
    </>
  );
};

export default AlgorithmHeader;
