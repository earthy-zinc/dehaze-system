import Waterfall from "@/components/Waterfall";
import { Card } from "antd";
import React from "react";

function generateImagePaths(start: number, end: number): string[] {
  const paths: string[] = [];
  for (let i = start; i <= end; i++) {
    // 格式化数字，保证两位数（如06而不是6）
    const formattedNumber = i.toString().padStart(3, "0");
    paths.push(`/src/assets/clean/${formattedNumber}.png`);
  }
  return paths;
}

export default function DatasetDetail() {
  const list = generateImagePaths(66, 120);

  return (
    <>
      <Card>
        <Waterfall list={list} />
      </Card>
    </>
  );
}
