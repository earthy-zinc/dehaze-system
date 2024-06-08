import { Algorithm } from "@/api/algorithm/model";

export default [
  {
    id: 1,
    name: "AlexNet",
    type: "ImageClassification",
    description: "A deep convolutional neural network for image classification",
    importPath: "models.alexnet",
    children: [
      {
        id: 2,
        name: "AlexNetV1",
        type: "Model",
        description: "The original version of AlexNet",
        status: 1,
        size: 245760000,
        path: "alexnet.pth",
      },
    ],
  },
  {
    id: 3,
    name: "VGG16",
    type: "ImageClassification",
    description:
      "A deep, large convolutional neural network for image recognition",
    importPath: "models.vgg16",
    children: [
      {
        id: 4,
        name: "VGG16WithPretrained",
        type: "Model",
        description: "VGG16 with pre-trained weights on ImageNet",
        status: 1,
        size: 531582720,
        path: "vgg16.pth",
      },
    ],
  },
  {
    id: 5,
    name: "ResNet50",
    type: "ImageClassification",
    description: "Deep residual network that won the ILSVRC 2015",
    importPath: "models/resnet50",
    children: [
      {
        id: 6,
        name: "ResNet50Pretrained",
        type: "Model",
        description: "ResNet50 with pre-training on ImageNet dataset",
        status: 1,
        size: 961267200,
        children: [],
      },
    ],
  },
  {
    id: 7,
    name: "InceptionV3",
    type: "ImageClassification",
    description:
      "Google's Inception architecture version 3 for image recognition tasks",
    importPath: "models/inceptionv3",
    children: [
      {
        id: 8,
        name: "InceptionV3Trained",
        type: "Model",
        description: "InceptionV3 model trained on a variety of image datasets",
        status: 1,
        size: 922933968,
        children: [],
      },
    ],
  },
  {
    id: 9,
    name: "YOLOv3",
    type: "ObjectDetection",
    description:
      "Real-time object detection system using a single neural network",
    importPath: "models/yolov3",
    children: [
      {
        id: 10,
        name: "YOLOv3Tiny",
        type: "Model",
        description:
          "Smaller version of YOLOv3 suitable for resource-constrained devices",
        status: 1,
        size: 242599424,
        children: [],
      },
    ],
  },
  {
    id: 11,
    name: "U-Net",
    type: "SemanticSegmentation",
    description:
      "Convolutional neural network for biomedical image segmentation",
    importPath: "models/unet",
    children: [
      {
        id: 12,
        name: "U-NetOriginal",
        type: "Model",
        description: "Original U-Net architecture for image segmentation tasks",
        status: 1,
        size: 335544320,
        children: [],
      },
    ],
  },
  {
    id: 13,
    name: "MobileNetV2",
    type: "ImageClassification",
    description:
      "Lightweight deep learning model for mobile and embedded vision applications",
    importPath: "models/mobilenetv2",
    children: [
      {
        id: 14,
        name: "MobileNetV2Quantized",
        type: "Model",
        description: "Quantized version of MobileNetV2 for efficient inference",
        status: 1,
        size: 16289280,
        children: [],
      },
    ],
  },
] as Algorithm[];
