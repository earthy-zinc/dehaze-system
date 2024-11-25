import AlgorithmAPI from "@/api/algorithm";
import { Algorithm, AlgorithmQuery } from "@/api/algorithm/model";
import EditModal from "@/pages/dataset/components/EditModal";
import { DeleteOutlined, EditOutlined } from "@ant-design/icons";
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

/**
 * 递归函数，用于遍历和清理 Dataset 结构
 * @param algorithms - 当前遍历的 Dataset 数组
 */
function cleanAlgorithm(algorithms: Algorithm[]): void {
  for (const element of algorithms) {
    const algorithm = element;

    if (algorithm.children && algorithm.children.length > 0) {
      // 递归遍历每个子数据集
      cleanAlgorithm(algorithm.children);
    } else if (
      algorithm.children &&
      Object.keys(algorithm.children).length === 0
    ) {
      delete algorithm.children;
    }
  }
}

export default function AlgorithmList(): React.JSX.Element {
  const [algorithmList, setAlgorithmList] = useState<Algorithm[]>();
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<AlgorithmQuery>({});

  const handleDelete = useCallback(
    (id: number) => {
      return () => {
        AlgorithmAPI.deleteByIds([id.toString()]).then(() => {
          const newAlgorithmList = algorithmList?.filter(
            (algorithm) => algorithm.id !== id
          );
          setAlgorithmList(newAlgorithmList);
        });
      };
    },
    [algorithmList]
  );

  const columns: TableColumnsType<Algorithm> = useMemo(
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
        minWidth: 250,
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
        title: "存储位置",
        dataIndex: "path",
        key: "path",
        width: 50,
        align: "center",
      },
      {
        title: "代码导入路径",
        dataIndex: "importPath",
        key: "importPath",
        width: 50,
        align: "center",
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
    [handleDelete]
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
    "importPath",
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
    const getAlgorithmInfo = () => {
      setLoading(true);
      AlgorithmAPI.getList(queryParams).then((data) => {
        cleanAlgorithm(data);
        setAlgorithmList(data);
        setLoading(false);
      });
    };
    getAlgorithmInfo();
  }, [queryParams]);

  const onFinish = (values: AlgorithmQuery) => {
    setQueryParams(values);
  };

  const onReset = () => {
    setQueryParams({});
  };

  // 获取对话框实例
  const editRef = useRef(null);

  // 打开对话框
  const openModal = (type: string, row: Algorithm) => {
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
          rowKey={(record: Algorithm) => record.id}
          dataSource={algorithmList}
          loading={loading}
          pagination={false}
        />
      </Card>

      <EditModal ref={editRef} />
    </div>
  );
}
