export interface FeatureItem {
  id: string;
  title: string;
  description: string;
  icon: string;
  color?: string;
}

export interface ToolItem {
  id: string;
  title: string;
  description: string;
  icon: string;
  target: string;
}

export interface SpecItem {
  icon: string;
  title: string;
  value: string;
  description: string;
}

export interface WorkflowStep {
  id: string;
  number: string;
  title: string;
  description: string;
  icon: string;
  target: string;
}

export const homeData = {
  // 工作流程数据
  workflowSteps: [
    {
      id: "image-input",
      number: "01",
      title: "图像输入",
      description: "支持上传、拍照、样例图片\n多种输入方式随心选择",
      icon: "camera",
      target: "image-input",
    },
    {
      id: "algorithm-select",
      number: "02",
      title: "智能算法",
      description: "多种去雾算法可选\nAI智能推荐最优方案",
      icon: "gift",
      target: "algorithm-select",
    },
    {
      id: "processing",
      number: "03",
      title: "一键处理",
      description: "毫秒级处理速度\n实时预览处理效果",
      icon: "play-circle-fill",
      target: "processing",
    },
  ] as WorkflowStep[],

  // 工具数据
  tools: [
    {
      id: "side-by-side",
      title: "并排对比",
      description: "多图并排展示，支持2-4张图片同屏对比",
      icon: "grid",
      target: "side-by-side",
    },
    {
      id: "overlay",
      title: "重叠对比",
      description: "拖动分割线实时对比，支持横向和纵向模式",
      icon: "photo",
      target: "overlay",
    },
    {
      id: "magnifier",
      title: "放大镜",
      description: "局部细节放大查看，精确对比图像质量",
      icon: "search",
      target: "magnifier",
    },
    {
      id: "filter",
      title: "滤镜调节",
      description: "实时调节亮度、对比度、饱和度等参数",
      icon: "setting",
      target: "filter",
    },
    {
      id: "metrics",
      title: "指标评估",
      description: "SSIM、PSNR等专业指标定量分析",
      icon: "integral",
      target: "metrics",
    },
    {
      id: "dataset",
      title: "数据集管理",
      description: "浏览和管理多个专业去雾数据集",
      icon: "server-fill",
      target: "dataset",
    },
  ] as ToolItem[],

  // 技术规格数据
  specs: [
    {
      icon: "play-circle-fill",
      title: "高性能",
      value: "60fps",
      description: "流畅运行，响应时间<200ms",
    },
    {
      icon: "grid",
      title: "全平台",
      value: "100%",
      description: "完美适配手机、平板、桌面",
    },
    {
      icon: "gift",
      title: "智能算法",
      value: "8+",
      description: "支持多种先进去雾算法",
    },
    {
      icon: "integral",
      title: "专业评估",
      value: "5+",
      description: "多维度定量分析指标",
    },
  ] as SpecItem[],

  // 算法特性数据
  algorithmFeatures: [
    "智能推荐最适合的去雾算法",
    "实时对比不同算法的处理效果",
    "毫秒级处理速度，即时查看结果",
    "支持批量处理和参数自定义",
  ],
};
