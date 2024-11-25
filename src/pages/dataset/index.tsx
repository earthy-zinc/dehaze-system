import DatasetAPI from "@/api/dataset";
import { Dataset, DatasetQuery } from "@/api/dataset/model";
import EditModal from "@/pages/dataset/components/EditModal";
import { DeleteOutlined, EditOutlined, EyeOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  Checkbox,
  Form,
  Input,
  Popconfirm,
  Popover,
  Table,
  TableColumnsType,
} from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate } from "react-router-dom";

/**
 * 递归函数，用于遍历和清理 Dataset 结构
 * @param datasets - 当前遍历的 Dataset 数组
 */
function cleanDatasets(datasets: Dataset[]): void {
  for (const element of datasets) {
    const dataset = element;

    if (dataset.children && dataset.children.length > 0) {
      // 递归遍历每个子数据集
      cleanDatasets(dataset.children);
    } else if (dataset.children && Object.keys(dataset.children).length === 0) {
      delete dataset.children;
    }
  }
}

export default function DatasetList() {
  const [datasetList, setDatasetList] = useState<Dataset[]>();
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<DatasetQuery>({});

  const navigate = useNavigate();
  const handleShow = useCallback(
    (id: number) => {
      return () => {
        navigate(`/dataset/${id}`);
      };
    },
    [navigate]
  );

  const handleDelete = useCallback(
    (id: number) => {
      return () => {
        DatasetAPI.deleteByIds([id.toString()]).then(() => {
          const newDatasetList = datasetList?.filter(
            (dataset) => dataset.id !== id
          );
          setDatasetList(newDatasetList);
        });
      };
    },
    [datasetList]
  );

  const columns: TableColumnsType<Dataset> = useMemo(
    () => [
      {
        title: "名称",
        dataIndex: "name",
        key: "name",
        width: 160,
        align: "center",
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 80,
        align: "center",
      },
      {
        title: "描述",
        dataIndex: "description",
        key: "description",
      },
      {
        title: "大小",
        dataIndex: "size",
        key: "size",
        width: 90,
        align: "center",
      },
      {
        title: "图片数量",
        dataIndex: "total",
        key: "total",
        width: 80,
        align: "center",
      },
      {
        title: "存储位置",
        dataIndex: "path",
        key: "path",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 50,
        align: "center",
        render: (status: number) => {
          return status === 1 ? "正常" : "异常";
        },
      },
      {
        title: "操作",
        key: "action",
        width: 180,
        align: "center",
        fixed: "right",
        render: (text: string, record: any) => (
          <>
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={handleShow(record.id)}
            >
              查看
            </Button>
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => openModal("新增", record)}
            >
              新增
            </Button>
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => openModal("编辑", record)}
            >
              编辑
            </Button>
            <Popconfirm
              title="确认要删除当前数据集吗？"
              onConfirm={handleDelete(record.id)}
              okText="确定"
              cancelText="取消"
            >
              <Button type="text" size="small" icon={<DeleteOutlined />} danger>
                删除
              </Button>
            </Popconfirm>
          </>
        ),
      },
    ],
    [handleDelete, handleShow]
  );

  const tableColumns = columns.map((column) => {
    return {
      label: column.title as string,
      value: column.key as string,
    };
  });

  const [selectedColumns, setSelectedColumns] = useState([
    "name",
    "type",
    "description",
    "size",
    "total",
    "path",
    "status",
    "action",
  ]);
  const handleColumnChange = (checkedValues: string[]) => {
    setSelectedColumns(checkedValues);
  };
  const tableSettings = (
    <Checkbox.Group
      options={tableColumns}
      defaultValue={selectedColumns}
      onChange={handleColumnChange}
    ></Checkbox.Group>
  );

  const visibleColumns = useMemo(() => {
    return columns.filter((column) =>
      selectedColumns.includes(column.key as string)
    );
  }, [columns, selectedColumns]);

  useEffect(() => {
    const getDatasetList = () => {
      setLoading(true);
      DatasetAPI.getList(queryParams).then((data) => {
        cleanDatasets(data);
        setDatasetList(data);
        setLoading(false);
      });
    };
    getDatasetList();
  }, [queryParams]);

  const onFinish = (values: DatasetQuery) => {
    setQueryParams(values);
  };

  const onReset = () => {
    setQueryParams({});
  };

  // 获取对话框实例
  const editRef = useRef(null);

  // 打开对话框
  const openModal = (type: string, row: Dataset) => {
    (editRef.current as any)?.open(type, row);
  };

  return (
    <div className="app-container">
      <Card className="search-container">
        <Form layout="inline" onFinish={onFinish} onReset={onReset}>
          <Form.Item name="keywords" label="关键字">
            <Input placeholder="数据集名称" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              搜索
            </Button>
          </Form.Item>
          <Form.Item>
            <Button type="default" htmlType="reset">
              重置
            </Button>
          </Form.Item>
          <Form.Item>
            <Popover content={tableSettings}>
              <Button type="default">设置</Button>
            </Popover>
          </Form.Item>
        </Form>
      </Card>

      <Card style={{ overflowX: "hidden" }}>
        <Table
          columns={visibleColumns}
          expandable={{
            defaultExpandAllRows: true,
            indentSize: 10,
          }}
          scroll={{ x: true }}
          rowKey={(record: Dataset) => record.id}
          dataSource={datasetList}
          loading={loading}
          pagination={false}
        />
      </Card>

      <EditModal ref={editRef} />
    </div>
  );
}
