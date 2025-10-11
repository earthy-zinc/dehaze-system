import DatasetAPI from "@/api/dataset";
import { Dataset, ImageItem } from "@/api/dataset/model";
import Waterfall from "@/components/Waterfall";
import { useWindowSize } from "@/hooks/useWindowSize";
import { Button, Card, Divider, Form, Input } from "antd";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";

interface ImageType {
  id: number;
  type: string;
  enabled: boolean;
}

const breakpoints = [
  { minWidth: 0, columns: 1 },
  { minWidth: 768, columns: 2 },
  { minWidth: 1024, columns: 3 },
  { minWidth: 1280, columns: 4 },
];

const uniqueArray = (arr: ImageItem[]) => {
  const seen = new Set();
  return arr.filter((item) => {
    if (!seen.has(item.id)) {
      seen.add(item.id);
      return true;
    }
    return false;
  });
};

const getImageList = (typeId: number, data: ImageItem[]) => {
  return data.map((item) => ({
    id: item.id,
    src: item.imgUrl[typeId].url,
    originSrc: item.imgUrl[typeId].originUrl,
    alt: item.imgUrl[typeId].description,
  }));
};

const getImageTypes = (data: ImageItem[]) => {
  return (
    data[0]?.imgUrl.map((item, index) => ({
      id: index,
      type: item.type,
      enabled: index === 0,
    })) || []
  );
};
export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  const datasetId = Number(id) || 0;
  const [queryParams, setQueryParams] = useState({
    pageNum: 1,
    pageSize: 10,
    keywords: "",
  });
  const [totalPages, setTotalPages] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);

  const [datasetInfo, setDatasetInfo] = useState<Dataset | null>(null);
  const [imageData, setImageData] = useState<ImageItem[]>([]);
  const [imageTypes, setImageTypes] = useState<ImageType[]>([]);
  const curImageType = useMemo(
    () => imageTypes.find((type) => type.enabled),
    [imageTypes]
  );
  const imageList = useMemo(() => {
    return getImageList(curImageType?.id ?? 0, imageData);
  }, [imageData, curImageType]);

  const { width } = useWindowSize();
  const loadingBarRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const itemWidth = useMemo(() => {
    return breakpoints.reduce(
      (acc, breakpoint) =>
        width >= breakpoint.minWidth
          ? Math.floor((width - 60) / breakpoint.columns)
          : acc,
      400
    );
  }, [width]);

  // 获取数据集信息
  useEffect(() => {
    let isMounted = true;

    if (isMounted) {
      DatasetAPI.getDatasetInfoById(datasetId).then((datasetInfo) => {
        setDatasetInfo(datasetInfo);
        setLoading(false);
      });
    }

    return () => {
      isMounted = false;
    };
  }, [datasetId]);

  // 获取图片数据
  useEffect(() => {
    DatasetAPI.getImageItem(datasetId, queryParams).then((data) => {
      if (queryParams.pageNum === 1) {
        setImageData(data.list);
        setImageTypes(getImageTypes(data.list));
        setTotalPages(Math.ceil(data.total / queryParams.pageSize));
      } else {
        setImageData((prev) => [...prev, ...data.list]);
      }
    });
  }, [datasetId, queryParams]);

  // 图片类型切换
  const handleImageTypeChange = (typeId: number) =>
    setImageTypes((prev) =>
      prev.map((item) => ({ ...item, enabled: item.id === typeId }))
    );

  // 设置观察器
  useEffect(() => {
    if (observerRef.current) observerRef.current.disconnect();
    observerRef.current = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && queryParams.pageNum < totalPages) {
          const nextPage = queryParams.pageNum + 1;
          console.log(nextPage);
          setQueryParams((prev) => ({ ...prev, pageNum: nextPage }));
        }
      });
    });

    if (loadingBarRef.current) {
      const loadingBarEl = loadingBarRef.current;
      loadingBarEl.style.transform = "translate3d(0, 3000px, 0)";
      observerRef.current.observe(loadingBarEl);
      setTimeout(() => (loadingBarEl.style.transform = "none"), 2000);
    }

    return () => observerRef.current?.disconnect();
  }, [queryParams.pageNum, totalPages]);

  // 重置查询
  const resetQuery = () => {
    setQueryParams({ pageNum: 1, pageSize: 10, keywords: "" });
  };

  // 搜索
  const handleSearch = () => {
    setQueryParams((prev) => ({ ...prev, pageNum: 1 }));
  };

  // 显示大图（根据需要实现）
  const showBigPicture = (itemId: number) => {
    // 实现图片放大功能
  };

  return (
    <div className="app-container">
      <Card>
        <h1 className="mt-2 mb-3" style={{ textAlign: "center" }}>
          {datasetInfo?.name} {datasetInfo?.type} 数据集
        </h1>
        <p className="mr-3 ml-3 mb-6" style={{ textIndent: "2em" }}>
          {datasetInfo?.description}
        </p>
        <div
          className="mb-1"
          style={{ display: "flex", justifyContent: "space-between" }}
        >
          <Button.Group>
            {imageTypes.map((imageType) => (
              <Button
                key={imageType.id}
                type={imageType.enabled ? "primary" : "default"}
                onClick={() => handleImageTypeChange(imageType.id)}
              >
                {imageType.type}
              </Button>
            ))}
          </Button.Group>
          <Form layout="inline">
            <Form.Item label="关键字">
              <Input
                value={queryParams.keywords}
                onChange={(e) =>
                  setQueryParams((prev) => ({
                    ...prev,
                    keywords: e.target.value,
                  }))
                }
                placeholder="图片名称"
                onPressEnter={handleSearch}
              />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleSearch}>
                搜索
              </Button>
              <Button onClick={resetQuery}>重置</Button>
            </Form.Item>
          </Form>
        </div>
        <Waterfall
          list={imageList}
          width={itemWidth}
          onClickItem={showBigPicture}
        />
        <div ref={loadingBarRef}>
          {queryParams.pageNum < totalPages && (
            <Divider>正在加载，请稍后</Divider>
          )}
        </div>
      </Card>
    </div>
  );
}
