import {
  MemberAPI,
  type GrowthLogVO,
  type MemberDetailVO,
  type MemberLevelCode,
} from "dehaze-sdk-js";
import {
  Descriptions,
  Drawer,
  Empty,
  Spin,
  Table,
  Tabs,
  Tag,
  type TableColumnsType,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useState,
} from "react";

const LEVEL_COLOR_MAP: Record<MemberLevelCode, string> = {
  level_0: "default",
  level_1: "blue",
  level_2: "purple",
  level_3: "gold",
};

const GROWTH_CHANGE_TYPE_LABEL: Record<string, string> = {
  dehaze: "去雾",
  evaluate: "评估",
  rating: "评分",
  sign_in: "签到",
  sign_in_bonus: "签到奖励",
  consume: "消费",
  refund_deduct: "退款扣减",
  admin_adjust: "管理员调整",
};

export interface MemberDetailDrawerRef {
  open: (userId: number) => void;
}

const MemberDetailDrawerImpl = forwardRef<
  MemberDetailDrawerRef,
  Record<string, never>
>((_, ref) => {
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [userId, setUserId] = useState(0);
  const [detail, setDetail] = useState<MemberDetailVO | undefined>(undefined);

  const [growthLogs, setGrowthLogs] = useState<GrowthLogVO[]>([]);
  const [growthLogTotal, setGrowthLogTotal] = useState(0);
  const [growthLogLoading, setGrowthLogLoading] = useState(false);
  const [growthLogPage, setGrowthLogPage] = useState({
    pageNum: 1,
    pageSize: 10,
  });

  const open = useCallback((id: number) => {
    setUserId(id);
    setVisible(true);
    setDetail(undefined);
    setGrowthLogs([]);
    setGrowthLogTotal(0);
    setGrowthLogPage({ pageNum: 1, pageSize: 10 });
  }, []);

  useImperativeHandle(ref, () => ({ open }), [open]);

  const loadDetail = useCallback(async (id: number) => {
    setLoading(true);
    try {
      const data = await MemberAPI.getDetail(id);
      setDetail(data);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadGrowthLogs = useCallback(
    async (page: { pageNum: number; pageSize: number }) => {
      setGrowthLogLoading(true);
      try {
        const result = await MemberAPI.getGrowthLogs(page);
        setGrowthLogs(result.list || []);
        setGrowthLogTotal(result.total || 0);
      } finally {
        setGrowthLogLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    if (visible && userId) {
      loadDetail(userId);
      loadGrowthLogs(growthLogPage);
    }
  }, [visible, userId, loadDetail, loadGrowthLogs, growthLogPage]);

  const handleGrowthLogPageChange = useCallback(
    (pageNum: number, pageSize: number) => {
      setGrowthLogPage({ pageNum, pageSize });
    },
    []
  );

  const growthLogColumns: TableColumnsType<GrowthLogVO> = [
    {
      title: "时间",
      dataIndex: "createTime",
      key: "createTime",
      width: 180,
      align: "center",
    },
    {
      title: "类型",
      dataIndex: "changeType",
      key: "changeType",
      width: 120,
      align: "center",
      render: (type: string) => (
        <Tag>{GROWTH_CHANGE_TYPE_LABEL[type] || type}</Tag>
      ),
    },
    {
      title: "变动",
      dataIndex: "changeValue",
      key: "changeValue",
      width: 100,
      align: "center",
      render: (value: number) => (
        <span style={{ color: value >= 0 ? "#52c41a" : "#f5222d" }}>
          {value >= 0 ? "+" : ""}
          {value}
        </span>
      ),
    },
    {
      title: "余额",
      dataIndex: "balance",
      key: "balance",
      width: 100,
      align: "center",
    },
    {
      title: "原因",
      dataIndex: "reason",
      key: "reason",
      ellipsis: true,
    },
  ];

  return (
    <Drawer
      title="会员详情"
      width={780}
      open={visible}
      onClose={() => setVisible(false)}
      destroyOnHidden
    >
      <Spin spinning={loading}>
        <Tabs
          items={[
            {
              key: "basic",
              label: "基本信息",
              children: detail ? (
                <Descriptions column={2} bordered size="small">
                  <Descriptions.Item label="用户名">
                    {detail.username}
                  </Descriptions.Item>
                  <Descriptions.Item label="昵称">
                    {detail.nickname}
                  </Descriptions.Item>
                  <Descriptions.Item label="等级">
                    <Tag color={LEVEL_COLOR_MAP[detail.levelCode]}>
                      {detail.levelName}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="成长值">
                    {detail.growthValue}
                  </Descriptions.Item>
                  <Descriptions.Item label="到期时间">
                    {detail.expireTime || "成长值维持"}
                  </Descriptions.Item>
                  <Descriptions.Item label="状态">
                    <Tag color={detail.status === 1 ? "success" : "error"}>
                      {detail.status === 1 ? "正常" : "冻结"}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="开通时间">
                    {detail.becomeMemberTime || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="累计消费">
                    {detail.totalConsumption}
                  </Descriptions.Item>
                  <Descriptions.Item label="本月去雾已用">
                    {detail.monthlyDehazeUsed} / {detail.monthlyDehazeQuota}
                  </Descriptions.Item>
                  <Descriptions.Item label="本月评估已用">
                    {detail.monthlyEvaluateUsed} / {detail.monthlyEvaluateQuota}
                  </Descriptions.Item>
                  {detail.frozenReason && (
                    <Descriptions.Item label="冻结原因" span={2}>
                      {detail.frozenReason}
                      {detail.frozenTime && `（${detail.frozenTime}）`}
                    </Descriptions.Item>
                  )}
                </Descriptions>
              ) : (
                <Empty description="暂无数据" />
              ),
            },
            {
              key: "growth",
              label: "成长值流水",
              children: (
                <Table
                  size="small"
                  columns={growthLogColumns}
                  dataSource={growthLogs}
                  rowKey="id"
                  loading={growthLogLoading}
                  scroll={{ x: 700 }}
                  pagination={{
                    current: growthLogPage.pageNum,
                    pageSize: growthLogPage.pageSize,
                    total: growthLogTotal,
                    showSizeChanger: true,
                    showTotal: (t) => `共 ${t} 条`,
                    onChange: handleGrowthLogPageChange,
                  }}
                />
              ),
            },
            {
              key: "log",
              label: "操作日志",
              children: <Empty description="暂无操作日志" />,
            },
          ]}
        />
      </Spin>
    </Drawer>
  );
});

MemberDetailDrawerImpl.displayName = "MemberDetailDrawer";

export default MemberDetailDrawerImpl as any;
