import DatasetAPI from "@/api/dataset";
import { Dataset, ImageItem } from "@/api/dataset/model";
import Waterfall from "@/components/Waterfall";
import { ViewCard } from "@/components/Waterfall/types";
import { useWindowSize } from "@/hooks/useWindowSize";
import { Button, Card, Divider, Form, Input } from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useParams } from "react-router-dom";

interface ImageType {
  id: number;
  type: string;
  enabled: boolean;
}

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  const [datasetId, setDatasetId] = useState<number>(Number(id));
  const [totalPages, setTotalPages] = useState<number>(1);
  const [queryParams, setQueryParams] = useState({
    pageNum: 1,
    pageSize: 10,
    keywords: "",
  });
  const [datasetInfo, setDatasetInfo] = useState<Dataset>({
    id: 0,
    parentId: 0,
    name: "未知",
    description: "未知描述",
    type: "类型",
    path: "",
    size: "0",
    total: 0,
  });
  const [images, setImages] = useState<ViewCard[]>([]);
  const [imageData, setImageData] = useState<ImageItem[]>([]);
  const [imageTypes, setImageTypes] = useState<ImageType[]>([
    { id: 0, type: "清晰图像", enabled: true },
    { id: 1, type: "有雾图像", enabled: false },
  ]);
  const [renderCount, setRenderCount] = useState<number>(0);
  const loadingBarRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const { width } = useWindowSize();
  const itemWidth = useMemo(() => {
    const breakpoints = [
      { minWidth: 0, columns: 1 },
      { minWidth: 768, columns: 2 },
      { minWidth: 1024, columns: 3 },
      { minWidth: 1280, columns: 4 },
    ];
    return breakpoints.reduce(
      (acc, breakpoint) =>
        width >= breakpoint.minWidth
          ? Math.floor((width - 60) / breakpoint.columns)
          : acc,
      400
    );
  }, [width]);

  const fetchDatasetInfo = useCallback(async () => {
    try {
      const data = await DatasetAPI.getDatasetInfoById(datasetId);
      setDatasetInfo(data);
      await handleQuery();
    } catch (err) {
      console.error("Failed to fetch dataset info:", err);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  const handleQuery = useCallback(async () => {
    try {
      setLoading(true);
      const data = await DatasetAPI.getImageItem(datasetId, queryParams);
      setImageData((prev) =>
        queryParams.pageNum === 1 ? data.list : [...prev, ...data.list]
      );
      setTotalPages(Math.ceil(data.total / queryParams.pageSize));
      if (data.list.length > 0) {
        const tempImageTypes = data.list[0].imgUrl.map((item, index) => ({
          id: index,
          type: item.type,
          enabled: index === 0,
        }));
        setImageTypes(tempImageTypes);
        switchImageUrl(0);
      }
    } catch (err) {
      console.error("Failed to fetch images:", err);
    } finally {
      setLoading(false);
    }
  }, [datasetId, queryParams]);

  const switchImageUrl = useCallback(
    (id: number) => {
      const newImages = imageData.map((item) => ({
        id: item.id,
        src: item.imgUrl[id].url.replace(/localhost/, "172.16.3.113"),
        originSrc: item.imgUrl[id].originUrl?.replace(
          /localhost/,
          "172.16.3.113"
        ),
        alt: item.imgUrl[id].description,
      }));
      setImages(newImages);
    },
    [imageData]
  );

  const handleImageTypeChange = useCallback(
    (typeId: number) => {
      setImageTypes((prev) =>
        prev.map((item) => ({ ...item, enabled: item.id === typeId }))
      );
      switchImageUrl(typeId);
    },
    [switchImageUrl]
  );

  const handleObserver = useCallback(() => {
    if (loadingBarRef.current && queryParams.pageNum < totalPages) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setQueryParams((prev) => ({ ...prev, pageNum: prev.pageNum + 1 }));
          }
        });
      });
      observer.observe(loadingBarRef.current);
      return () => observer.disconnect();
    }
  }, [queryParams.pageNum, totalPages]);

  useEffect(() => {
    setLoading(true);
    fetchDatasetInfo();
  }, [fetchDatasetInfo]);

  useEffect(() => {
    handleObserver();
  }, [handleObserver]);

  const resetQuery = useCallback(() => {
    setQueryParams({ pageNum: 1, pageSize: 10, keywords: "" });
  }, []);

  const showBigPicture = (itemId: number) => {
    const curImageItem = imageData.find((item) => item.id === itemId);
    const result: string[] =
      curImageItem?.imgUrl.map((item) =>
        item.originUrl
          ? item.originUrl.replace(/localhost/, "172.16.3.113")
          : item.url.replace(/localhost/, "172.16.3.113")
      ) || [];
    // viewerApi({ images: result }).show().update().view(0);
  };

  return (
    <div className="app-container">
      <Card>
        <h1 className="mt-2 mb-3" style={{ textAlign: "center" }}>
          {datasetInfo.name} {datasetInfo.type} 数据集
        </h1>
        <p className="mr-3 ml-3 mb-6" style={{ textIndent: "2em" }}>
          {datasetInfo.description}
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
                  setQueryParams({ ...queryParams, keywords: e.target.value })
                }
                placeholder="图片名称"
                onPressEnter={handleQuery}
              />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleQuery}>
                搜索
              </Button>
              <Button onClick={resetQuery}>重置</Button>
            </Form.Item>
          </Form>
        </div>
        <Waterfall
          list={images}
          width={itemWidth}
          onClickItem={showBigPicture}
          onAfterRender={() => setRenderCount((prev) => prev + 1)}
        />
        <div ref={loadingBarRef}>
          {totalPages > 1 &&
            renderCount >= queryParams.pageNum - 1 &&
            queryParams.pageNum < totalPages && (
              <Divider>正在加载，请稍后</Divider>
            )}
        </div>
      </Card>
    </div>
  );
}
