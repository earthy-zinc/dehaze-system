import {
  MemberAPI,
  type BenefitForm,
  type BenefitVO,
  type MemberLevelCode,
} from "dehaze-sdk-js";
import {
  Button,
  Drawer,
  InputNumber,
  Spin,
  Switch,
  Table,
  Tag,
  message,
  type TableColumnsType,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const LEVEL_COLOR_MAP: Record<MemberLevelCode, string> = {
  level_0: "default",
  level_1: "blue",
  level_2: "purple",
  level_3: "gold",
};

export interface BenefitConfigDrawerRef {
  open: () => void;
}

const BenefitConfigDrawer = forwardRef<
  BenefitConfigDrawerRef,
  Record<string, never>
>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [benefitList, setBenefitList] = useState<BenefitVO[]>([]);

  const open = useCallback(() => {
    setVisible(true);
    setLoading(true);
    MemberAPI.listBenefits()
      .then((data) => {
        setBenefitList(data || []);
      })
      .catch((error) => {
        message.error(error?.message || "加载权益配置失败");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  useImperativeHandle(ref, () => ({ open }), [open]);

  const updateRow = useCallback(
    (levelCode: MemberLevelCode, field: keyof BenefitVO, value: number) => {
      setBenefitList((prev) =>
        prev.map((item) =>
          item.levelCode === levelCode ? { ...item, [field]: value } : item
        )
      );
    },
    []
  );

  const handleSave = useCallback((row: BenefitVO) => {
    const payload: BenefitForm = {
      levelName: row.levelName,
      growthMin: row.growthMin,
      growthMax: row.growthMax,
      monthlyDehazeQuota: row.monthlyDehazeQuota,
      monthlyEvaluateQuota: row.monthlyEvaluateQuota,
      historyRetention: row.historyRetention,
      batchLimit: row.batchLimit,
      priority: row.priority,
      advancedParams: row.advancedParams,
      hdExport: row.hdExport,
      reportExport: row.reportExport,
      batchDownload: row.batchDownload,
      sort: row.sort,
      status: row.status,
    };
    MemberAPI.updateBenefit(row.levelCode, payload)
      .then(() => {
        message.success(`「${row.levelName}」权益配置已保存`);
      })
      .catch((error) => {
        message.error(error?.message || "保存失败");
      });
  }, []);

  const columns: TableColumnsType<BenefitVO> = [
    {
      title: "等级",
      dataIndex: "levelCode",
      key: "levelCode",
      width: 110,
      align: "center",
      render: (levelCode: MemberLevelCode, record: BenefitVO) => (
        <Tag color={LEVEL_COLOR_MAP[levelCode]}>{record.levelName}</Tag>
      ),
    },
    {
      title: "月去雾配额",
      dataIndex: "monthlyDehazeQuota",
      key: "monthlyDehazeQuota",
      width: 120,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <InputNumber
          min={0}
          size="small"
          style={{ width: 100 }}
          value={val}
          onChange={(v) =>
            updateRow(record.levelCode, "monthlyDehazeQuota", Number(v) || 0)
          }
        />
      ),
    },
    {
      title: "月评估配额",
      dataIndex: "monthlyEvaluateQuota",
      key: "monthlyEvaluateQuota",
      width: 120,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <InputNumber
          min={0}
          size="small"
          style={{ width: 100 }}
          value={val}
          onChange={(v) =>
            updateRow(record.levelCode, "monthlyEvaluateQuota", Number(v) || 0)
          }
        />
      ),
    },
    {
      title: "历史保留",
      dataIndex: "historyRetention",
      key: "historyRetention",
      width: 110,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <InputNumber
          min={0}
          size="small"
          style={{ width: 90 }}
          value={val}
          onChange={(v) =>
            updateRow(record.levelCode, "historyRetention", Number(v) || 0)
          }
        />
      ),
    },
    {
      title: "批量上限",
      dataIndex: "batchLimit",
      key: "batchLimit",
      width: 110,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <InputNumber
          min={0}
          size="small"
          style={{ width: 90 }}
          value={val}
          onChange={(v) =>
            updateRow(record.levelCode, "batchLimit", Number(v) || 0)
          }
        />
      ),
    },
    {
      title: "优先级",
      dataIndex: "priority",
      key: "priority",
      width: 100,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <InputNumber
          min={0}
          size="small"
          style={{ width: 80 }}
          value={val}
          onChange={(v) =>
            updateRow(record.levelCode, "priority", Number(v) || 0)
          }
        />
      ),
    },
    {
      title: "高级参数",
      dataIndex: "advancedParams",
      key: "advancedParams",
      width: 110,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <InputNumber
          min={0}
          size="small"
          style={{ width: 90 }}
          value={val}
          onChange={(v) =>
            updateRow(record.levelCode, "advancedParams", Number(v) || 0)
          }
        />
      ),
    },
    {
      title: "高清导出",
      dataIndex: "hdExport",
      key: "hdExport",
      width: 90,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <Switch
          checked={val === 1}
          onChange={(checked) =>
            updateRow(record.levelCode, "hdExport", checked ? 1 : 0)
          }
        />
      ),
    },
    {
      title: "报告导出",
      dataIndex: "reportExport",
      key: "reportExport",
      width: 90,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <Switch
          checked={val === 1}
          onChange={(checked) =>
            updateRow(record.levelCode, "reportExport", checked ? 1 : 0)
          }
        />
      ),
    },
    {
      title: "批量下载",
      dataIndex: "batchDownload",
      key: "batchDownload",
      width: 90,
      align: "center",
      render: (val: number, record: BenefitVO) => (
        <Switch
          checked={val === 1}
          onChange={(checked) =>
            updateRow(record.levelCode, "batchDownload", checked ? 1 : 0)
          }
        />
      ),
    },
    {
      title: "操作",
      key: "action",
      width: 80,
      align: "center",
      fixed: "right",
      render: (_: unknown, record: BenefitVO) => (
        <Button type="link" size="small" onClick={() => handleSave(record)}>
          保存
        </Button>
      ),
    },
  ];

  return (
    <Drawer
      title="权益配置"
      width={1100}
      open={visible}
      onClose={() => setVisible(false)}
      destroyOnHidden
    >
      <Spin spinning={loading}>
        <Table
          columns={columns}
          dataSource={benefitList}
          rowKey="levelCode"
          scroll={{ x: 1200 }}
          pagination={false}
        />
      </Spin>
    </Drawer>
  );
});

BenefitConfigDrawer.displayName = "BenefitConfigDrawer";

export default BenefitConfigDrawer;
