import {
  MemberAPI,
  type MemberLevelCode,
  type MemberPageVO,
  type MemberQuery,
  type MemberStatus,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import {
  ArrowUpOutlined,
  CrownOutlined,
  EditOutlined,
  LockOutlined,
  ReloadOutlined,
  SearchOutlined,
  StarOutlined,
  UnlockOutlined,
} from "@ant-design/icons";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import BenefitConfigDrawer, {
  type BenefitConfigDrawerRef,
} from "./components/BenefitConfigDrawer";
import FreezeDialog, { type FreezeDialogRef } from "./components/FreezeDialog";
import GrowthAdjustDialog, {
  type GrowthAdjustDialogRef,
} from "./components/GrowthAdjustDialog";
import LevelAdjustDialog, {
  type LevelAdjustDialogRef,
} from "./components/LevelAdjustDialog";
import MemberDetailDrawer, {
  type MemberDetailDrawerRef,
} from "./components/MemberDetailDrawer";
import "./index.scss";

const { RangePicker } = DatePicker;

const LEVEL_COLOR_MAP: Record<MemberLevelCode, string> = {
  level_0: "default",
  level_1: "blue",
  level_2: "purple",
  level_3: "gold",
};

const LEVEL_OPTIONS: { label: string; value: MemberLevelCode }[] = [
  { label: "普通会员", value: "level_0" },
  { label: "高级会员", value: "level_1" },
  { label: "VIP会员", value: "level_2" },
  { label: "SVIP会员", value: "level_3" },
];

const STATUS_OPTIONS = [
  { label: "正常", value: 1 },
  { label: "冻结", value: 0 },
];

const MemberManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<MemberPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<MemberQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  const detailDrawerRef = useRef<MemberDetailDrawerRef>(null);
  const levelDialogRef = useRef<LevelAdjustDialogRef>(null);
  const growthDialogRef = useRef<GrowthAdjustDialogRef>(null);
  const freezeDialogRef = useRef<FreezeDialogRef>(null);
  const benefitDrawerRef = useRef<BenefitConfigDrawerRef>(null);

  const hasPerm = useHasPerm();

  const loadData = useCallback(async (params: MemberQuery) => {
    setLoading(true);
    try {
      const result = await MemberAPI.getPage(params);
      setPageData(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const handleSearch = useCallback(
    (values: {
      keywords?: string;
      levelCode?: MemberLevelCode;
      status?: MemberStatus;
      expireTimeRange?: [string, string];
      growthMin?: number;
      growthMax?: number;
    }) => {
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        keywords: values.keywords || undefined,
        levelCode: values.levelCode,
        status: values.status,
        expireTimeStart: values.expireTimeRange?.[0],
        expireTimeEnd: values.expireTimeRange?.[1],
        growthMin: values.growthMin,
        growthMax: values.growthMax,
      }));
    },
    []
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleDetail = useCallback((record: MemberPageVO) => {
    detailDrawerRef.current?.open(record.userId);
  }, []);

  const handleLevelAdjust = useCallback((record: MemberPageVO) => {
    levelDialogRef.current?.open(record);
  }, []);

  const handleGrowthAdjust = useCallback((record: MemberPageVO) => {
    growthDialogRef.current?.open(record);
  }, []);

  const handleFreeze = useCallback((record: MemberPageVO) => {
    freezeDialogRef.current?.open(record);
  }, []);

  const handleBenefitConfig = useCallback(() => {
    benefitDrawerRef.current?.open();
  }, []);

  const columns: TableColumnsType<MemberPageVO> = useMemo(
    () => [
      {
        title: "用户名",
        dataIndex: "username",
        key: "username",
        width: 140,
        align: "center",
      },
      {
        title: "昵称",
        dataIndex: "nickname",
        key: "nickname",
        width: 140,
        align: "center",
      },
      {
        title: "等级",
        dataIndex: "levelCode",
        key: "levelCode",
        width: 110,
        align: "center",
        render: (levelCode: MemberLevelCode, record: MemberPageVO) => (
          <Tag color={LEVEL_COLOR_MAP[levelCode]}>{record.levelName}</Tag>
        ),
      },
      {
        title: "成长值",
        dataIndex: "growthValue",
        key: "growthValue",
        width: 100,
        align: "center",
      },
      {
        title: "本月已用",
        dataIndex: "monthlyUsed",
        key: "monthlyUsed",
        width: 100,
        align: "center",
      },
      {
        title: "到期时间",
        dataIndex: "expireTime",
        key: "expireTime",
        width: 180,
        align: "center",
        render: (expireTime?: string) =>
          expireTime || <Tag bordered={false}>成长值维持</Tag>,
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 90,
        align: "center",
        render: (status: number) => (
          <Tag color={status === 1 ? "success" : "error"}>
            {status === 1 ? "正常" : "冻结"}
          </Tag>
        ),
      },
      {
        title: "开通时间",
        dataIndex: "becomeMemberTime",
        key: "becomeMemberTime",
        width: 180,
        align: "center",
        render: (time?: string) => time || "-",
      },
      {
        title: "操作",
        key: "action",
        width: 320,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: MemberPageVO) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              onClick={() => handleDetail(record)}
            >
              详情
            </Button>
            {hasPerm("member:level:edit") && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleLevelAdjust(record)}
              >
                等级
              </Button>
            )}
            {hasPerm("member:growth:edit") && (
              <Button
                type="link"
                size="small"
                icon={<ArrowUpOutlined />}
                onClick={() => handleGrowthAdjust(record)}
              >
                成长值
              </Button>
            )}
            {hasPerm("member:status:edit") && (
              <Button
                type="link"
                size="small"
                danger={record.status === 1}
                icon={
                  record.status === 1 ? <LockOutlined /> : <UnlockOutlined />
                }
                onClick={() => handleFreeze(record)}
              >
                {record.status === 1 ? "冻结" : "解冻"}
              </Button>
            )}
          </Space>
        ),
      },
    ],
    [hasPerm, handleDetail, handleLevelAdjust, handleGrowthAdjust, handleFreeze]
  );

  return (
    <div className="member-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="用户名/昵称"
              allowClear
              style={{ width: 180 }}
            />
          </Form.Item>
          <Form.Item name="levelCode" label="等级">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={LEVEL_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 100 }}
              options={STATUS_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="expireTimeRange" label="到期时间">
            <RangePicker style={{ width: 240 }} />
          </Form.Item>
          <Form.Item label="成长值">
            <Space>
              <Form.Item name="growthMin" noStyle>
                <InputNumber
                  placeholder="最小"
                  style={{ width: 100 }}
                  min={0}
                />
              </Form.Item>
              <span>~</span>
              <Form.Item name="growthMax" noStyle>
                <InputNumber
                  placeholder="最大"
                  style={{ width: 100 }}
                  min={0}
                />
              </Form.Item>
            </Space>
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

      <Card
        className="table-card"
        size="small"
        title={
          hasPerm("member:benefit:edit") ? (
            <Button
              type="primary"
              icon={<StarOutlined />}
              onClick={handleBenefitConfig}
            >
              权益配置
            </Button>
          ) : null
        }
        extra={
          <Button icon={<CrownOutlined />} disabled>
            导出
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={pageData}
          rowKey="userId"
          loading={loading}
          scroll={{ x: 1300 }}
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

      <MemberDetailDrawer ref={detailDrawerRef} />
      <LevelAdjustDialog ref={levelDialogRef} onSuccess={refreshList} />
      <GrowthAdjustDialog ref={growthDialogRef} onSuccess={refreshList} />
      <FreezeDialog ref={freezeDialogRef} onSuccess={refreshList} />
      <BenefitConfigDrawer ref={benefitDrawerRef} />
    </div>
  );
};

export default MemberManagement;
