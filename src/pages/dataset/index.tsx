import DatasetAPI from "@/api/dataset";
import { Dataset, DatasetQuery } from "@/api/dataset/model";
import { DeleteOutlined, EditOutlined, EyeOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  Form,
  Input,
  Modal,
  Popconfirm,
  Table,
  TableColumnsType,
} from "antd";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function DatasetList() {
  const [datasetList, setDatasetList] = useState<Dataset[]>();
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<DatasetQuery>({});
  const [editModalVisible, setEditModalVisible] = useState(false);

  const columns: TableColumnsType<Dataset> = [
    {
      title: "名称",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "类型",
      dataIndex: "type",
      key: "type",
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
    },
    {
      title: "图片数量",
      dataIndex: "total",
      key: "total",
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
      render: (status: number) => {
        return status === 1 ? "正常" : "异常";
      },
    },
    {
      title: "操作",
      key: "action",
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
            onClick={handleEdit(record)}
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
  ];

  const navigate = useNavigate();

  useEffect(() => {
    const getDatasetList = () => {
      setLoading(true);
      DatasetAPI.getList(queryParams).then((data) => {
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

  const handleShow = (id: number) => {
    return () => {
      navigate(`/dataset/${id}`);
    };
  };

  const handleEdit = (record: Dataset) => {
    return () => {
      setEditModalVisible(true);
    };
  };

  const handleDelete = (id: number) => {
    return () => {
      DatasetAPI.deleteByIds([id.toString()]).then(() => {
        const newDatasetList = datasetList?.filter(
          (dataset) => dataset.id !== id
        );
        setDatasetList(newDatasetList);
      });
    };
  };

  const submitEdit = (values: Dataset) => {};

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
        </Form>
      </Card>
      <Card style={{ overflowX: "hidden" }}>
        <Table
          columns={columns}
          expandable={{
            defaultExpandAllRows: true,
          }}
          rowKey={(record: Dataset) => record.id}
          dataSource={datasetList}
          loading={loading}
          pagination={false}
        />
      </Card>
      <Modal
        title={"编辑数据集信息"}
        open={editModalVisible}
        okText="保存"
        cancelText="取消"
        okButtonProps={{ autoFocus: true, htmlType: "submit" }}
        onCancel={() => setEditModalVisible(false)}
        destroyOnClose
        modalRender={(dom) => (
          <Form
            layout="vertical"
            name="编辑数据集信息"
            onFinish={(values) => submitEdit(values)}
          >
            {dom}
          </Form>
        )}
      >
        <Form.Item>
          <Input></Input>
        </Form.Item>
      </Modal>
    </div>
  );
}
