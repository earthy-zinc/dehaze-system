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
      description: "快速处理图像\n实时预览处理效果",
      icon: "play-circle-fill",
      target: "processing",
    },
  ] as WorkflowStep[],

  // 工具数据（首页快捷入口，跳转 L2 功能页）
  tools: [
    {
      id: "image-input",
      title: "图像输入",
      description: "支持上传、拍照、样例图片多种输入方式",
      icon: "camera",
      target: "image-input",
    },
    {
      id: "algorithm",
      title: "算法库",
      description: "浏览多种去雾算法，智能推荐最优方案",
      icon: "gift",
      target: "algorithm",
    },
    {
      id: "dataset",
      title: "数据集",
      description: "浏览和管理多个专业去雾数据集",
      icon: "server-fill",
      target: "dataset",
    },
    {
      id: "metrics",
      title: "指标管理",
      description: "PSNR、SSIM等专业指标定量分析",
      icon: "integral",
      target: "metrics",
    },
  ] as ToolItem[],

  // 技术规格数据
  specs: [
    {
      icon: "play-circle-fill",
      title: "高性能",
      value: "GPU加速",
      description: "CUDA深度学习推理加速",
    },
    {
      icon: "grid",
      title: "全平台",
      value: "多端",
      description: "支持H5、小程序、App多端访问",
    },
    {
      icon: "gift",
      title: "智能算法",
      value: "",
      description: "支持多种先进去雾算法",
    },
    {
      icon: "integral",
      title: "专业评估",
      value: "4项",
      description: "PSNR、SSIM、MSE、FSIM定量指标",
    },
  ] as SpecItem[],

  // 算法特性数据
  algorithmFeatures: [
    "智能推荐最适合的去雾算法",
    "实时对比不同算法的处理效果",
    "GPU加速推理，快速查看结果",
    "支持参数自定义和效果评估",
  ],
};
