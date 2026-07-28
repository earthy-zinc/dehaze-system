import {
  MessageTemplateAPI,
  type MessageTemplateQuery,
  type MessageTemplateVO,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  EditOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
  message,
} from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import MessageTemplateFormDialog, {
  type MessageTemplateFormDialogRef,
} from "./components/MessageTemplateFormDialog";
import "./index.scss";

const TYPE_OPTIONS = [
  { value: "inbox", label: "站内信" },
  { value: "announcement", label: "系统公告" },
  { value: "business", label: "业务通知" },
  { value: "member", label: "会员通知" },
  { value: "alert", label: "告警通知" },
  { value: "critical_alert", label: "严重告警" },
];

const TYPE_LABEL: Record<string, string> = Object.fromEntries(
  TYPE_OPTIONS.map((o) => [o.value, o.label])
);

const PRIORITY_LABEL: Record<number, string> = {
  1: "低",
  2: "中",
  3: "高",
  4: "紧急",
};

const PRIORITY_COLOR: Record<number, string> = {
  1: "default",
  2: "blue",
  3: "orange",
  4: "red",
};

const CHANNEL_LABEL: Record<string, string> = {
  inbox: "站内信",
  push: "APP 推送",
  email: "邮件",
};

function formatChannels(channels?: Record<string, boolean>) {
  if (!channels) return "-";
  return Object.entries(channels)
    .filter(([, v]) => v)
    .map(([k]) => CHANNEL_LABEL[k] ?? k)
    .join("、");
}

const MessageTemplateManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<MessageTemplateVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<MessageTemplateQuery>({
    pageNum: 1,
    pageSize: 20,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detailData, setDetailData] = useState<MessageTemplateVO | null>(null);

  const formDialogRef = useRef<MessageTemplateFormDialogRef>(null);
  const hasPerm = useHasPerm();

  const loadData = useCallback((params: MessageTemplateQuery) => {
    setLoading(true);
    MessageTemplateAPI.getPage(params)
      .then((result) => {
        setPageData(result.list || []);
        setTotal(result.total || 0);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const handleSearch = useCallback(
    (values: { name?: string; type?: string; status?: number }) => {
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        name: values.name || undefined,
        type: values.type || undefined,
        status: values.status,
      }));
    },
    []
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 20 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleEdit = useCallback((record: MessageTemplateVO) => {
    formDialogRef.current?.open(record.id);
  }, []);

  const openDetail = useCallback((record: MessageTemplateVO) => {
    setLoading(true);
    MessageTemplateAPI.getDetail(record.id)
      .then((data) => {
        setDetailData(data);
        setDetailVisible(true);
      })
      .catch((err) => message.error(err?.message || "获取详情失败"))
      .finally(() => setLoading(false));
  }, []);

  const columns: TableColumnsType<MessageTemplateVO> = useMemo(
    () => [
      {
        title: "模板编码",
        dataIndex: "code",
        key: "code",
        width: 180,
        ellipsis: true,
      },
      {
        title: "模板名称",
        dataIndex: "name",
        key: "name",
        minWidth: 160,
        ellipsis: true,
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 110,
        align: "center",
        render: (type: string) => (
          <span className={`type-tag tag-${type}`}>
            {TYPE_LABEL[type] ?? type}
          </span>
        ),
      },
      {
        title: "标题模板",
        dataIndex: "titleTemplate",
        key: "titleTemplate",
        minWidth: 200,
        ellipsis: true,
      },
      {
        title: "优先级",
        dataIndex: "priority",
        key: "priority",
        width: 90,
        align: "center",
        render: (p: number) => (
          <Tag color={PRIORITY_COLOR[p] ?? "default"} bordered={false}>
            {PRIORITY_LABEL[p] ?? String(p)}
          </Tag>
        ),
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 90,
        align: "center",
        render: (s: number) =>
          s === 1 ? (
            <Tag color="success" bordered={false}>
              启用
            </Tag>
          ) : (
            <Tag bordered={false}>禁用</Tag>
          ),
      },
      {
        title: "更新时间",
        dataIndex: "updateTime",
        key: "updateTime",
        width: 170,
        align: "center",
        render: (t?: string, record?: MessageTemplateVO) =>
          t || record?.createTime || "-",
      },
      {
        title: "操作",
        key: "action",
        width: 160,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: MessageTemplateVO) => (
          <Space size="small">
            {hasPerm("notify:template:edit") && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEdit(record)}
              >
                编辑
              </Button>
            )}
            <Button type="link" size="small" onClick={() => openDetail(record)}>
              详情
            </Button>
          </Space>
        ),
      },
    ],
    [hasPerm, handleEdit, openDetail]
  );

  return (
    <div className="message-template-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="name" label="模板名称">
            <Input placeholder="模板名称" allowClear style={{ width: 200 }} />
          </Form.Item>
          <Form.Item name="type" label="消息类型">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={TYPE_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 120 }}
              options={[
                { value: 1, label: "启用" },
                { value: 0, label: "禁用" },
              ]}
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SearchOutlined />}
              >
                搜索
              </Button>
              <Button
                htmlType="reset"
                icon={<ReloadOutlined />}
                onClick={handleReset}
              >
                重置
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={pageData}
          rowKey={(record) => record.id}
          loading={loading}
          scroll={{ x: 1200 }}
          pagination={{
            current: queryParams.pageNum,
            pageSize: queryParams.pageSize,
            total,
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: ["10", "20", "50", "100"],
            showTotal: (t) => `共 ${t} 条`,
            onChange: handlePageChange,
          }}
        />
      </Card>

      <MessageTemplateFormDialog ref={formDialogRef} onSuccess={refreshList} />

      <Modal
        title="模板详情"
        open={detailVisible}
        width={680}
        footer={null}
        onCancel={() => setDetailVisible(false)}
        destroyOnHidden
      >
        {detailData && (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="模板编码">
              {detailData.code}
            </Descriptions.Item>
            <Descriptions.Item label="模板名称">
              {detailData.name}
            </Descriptions.Item>
            <Descriptions.Item label="消息类型">
              {TYPE_LABEL[detailData.type] ?? detailData.type}
            </Descriptions.Item>
            <Descriptions.Item label="标题模板">
              <code className="template-code">{detailData.titleTemplate}</code>
            </Descriptions.Item>
            {detailData.contentTemplate && (
              <Descriptions.Item label="正文模板">
                <pre className="template-pre">{detailData.contentTemplate}</pre>
              </Descriptions.Item>
            )}
            <Descriptions.Item label="优先级">
              {PRIORITY_LABEL[detailData.priority] ?? detailData.priority}
            </Descriptions.Item>
            <Descriptions.Item label="推送渠道">
              {formatChannels(detailData.channels)}
            </Descriptions.Item>
            {detailData.variables?.length ? (
              <Descriptions.Item label="模板变量">
                <div className="variable-list">
                  {detailData.variables.map((v) => (
                    <div className="variable-item" key={v.name}>
                      <code>{`{${v.name}}`}</code>
                      <span className="variable-desc">{v.desc}</span>
                    </div>
                  ))}
                </div>
              </Descriptions.Item>
            ) : null}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default MessageTemplateManagement;
