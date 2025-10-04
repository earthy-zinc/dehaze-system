import DatasetAPI from "@/api/dataset";
import { Dataset, ImageItemQuery } from "@/api/dataset/model";
import Waterfall from "@/components/Waterfall";
import { ViewCard } from "@/components/Waterfall/types";
import { changeUrl } from "@/utils";
import { Button, Card, Form, TreeSelect } from "antd";
import React, { useEffect, useRef, useState } from "react";

interface ImageType {
  id: number;
  type: string;
  enabled: boolean;
}

interface DatasetImageSelectProps {
  onSelected: (haze: string, clean: string) => void;
}

const DatasetImageSelect: React.FC<DatasetImageSelectProps> = ({
  onSelected,
}) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState<number>(1);
  const [datasetOptions, setDatasetOptions] = useState<OptionType[]>([]);
  const [datasetInfo, setDatasetInfo] = useState<Dataset>({
    id: 0,
    parentId: 0,
    name: "",
    type: "",
    description: "",
    createTime: new Date(),
    updateTime: new Date(),
    path: "",
    size: "",
    total: 0,
  });
  const [imageTypes, setImageTypes] = useState<ImageType[]>([]);
  const [images, setImages] = useState<ViewCard[]>([]);
  const [queryParams, setQueryParams] = useState<ImageItemQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const loadingBarRef = useRef<HTMLDivElement>(null);
  const observer = useRef<IntersectionObserver | null>(null);

  // 获取数据集可选项，并默认选择第一个数据集展示
  useEffect(() => {
    const fetchData = async () => {
      const options = await DatasetAPI.getOptions();
      setDatasetOptions(options);
      await handleSelectDataset();
    };
    fetchData().then();
  }, []);

  // 获取数据集信息并初始化数据
  const handleSelectDataset = async () => {
    const data = await DatasetAPI.getDatasetInfoById(selectedDatasetId);
    setDatasetInfo(data);
    await handleQuery();
  };

  // 请求当前数据集的图片列表
  const handleQuery = async () => {
    const response = await DatasetAPI.getImageItem(
      selectedDatasetId,
      queryParams
    );
    const items = response.list;
    const totalPages = Math.ceil(response.total / queryParams.pageSize);

    const currentType =
      imageTypes.find((type) => type.enabled) || imageTypes[0];
    const newImages = items.map((item) => ({
      id: item.id,
      src: changeUrl(item.imgUrl[currentType.id].url),
      originSrc: changeUrl(item.imgUrl[currentType.id].originUrl!),
      alt: item.imgUrl[currentType.id].description || "",
    }));
    setImages(newImages);
    // 更新分页状态
  };

  const handleImageTypeChange = (typeId: number) => {
    const newTypes = imageTypes.map((type) => ({
      ...type,
      enabled: type.id === typeId,
    }));
    setImageTypes(newTypes);
    // 重新渲染图片
  };

  const handleDatasetChange = (value: number) => {
    setSelectedDatasetId(value);
  };

  // 实现分页加载
  useEffect(() => {
    const setupObserver = () => {
      if (!loadingBarRef.current) return;
      observer.current = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setQueryParams((prev) => ({ ...prev, pageNum: prev.pageNum + 1 }));
            handleQuery();
          }
        });
      });
      observer.current.observe(loadingBarRef.current);
      return () => observer.current?.disconnect();
    };
    setupObserver();
  }, [queryParams.pageNum]);

  return (
    <Card style={{ overflowY: "scroll", height: "73vh" }}>
      <Form layout="vertical">
        <Form.Item label="选择数据集">
          <TreeSelect
            value={selectedDatasetId}
            treeData={datasetOptions}
            onChange={handleDatasetChange}
            treeDataSimpleMode
          />
        </Form.Item>
      </Form>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <div>
          {imageTypes.map((type) => (
            <Button
              key={type.id}
              type={type.enabled ? "primary" : "default"}
              onClick={() => handleImageTypeChange(type.id)}
            >
              {type.type}
            </Button>
          ))}
        </div>
        {/* 搜索表单 */}
      </div>
      <Waterfall list={images} />
      <div ref={loadingBarRef} style={{ textAlign: "center", padding: 8 }}>
        正在加载...
      </div>
    </Card>
  );
};

export default DatasetImageSelect;
