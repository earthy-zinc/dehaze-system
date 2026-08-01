import {
  AlgorithmAPI,
  RecommendationAPI,
  type OptionType,
  type RecommendationRule,
} from "dehaze-sdk-js";
import {
  PlusOutlined,
  SaveOutlined,
  ReloadOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  type TableColumnsType,
} from "antd";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import "./index.module.scss";

export default function RecommendationRules() {
  const [loading, setLoading] = useState(false);
  const [rules, setRules] = useState<RecommendationRule[]>([]);
  const [algorithmOptions, setAlgorithmOptions] = useState<OptionType[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingRule, setEditingRule] = useState<RecommendationRule | null>(
    null
  );
  const [form] = Form.useForm();

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [ruleData, algoData] = await Promise.all([
        RecommendationAPI.getRules(),
        AlgorithmAPI.getOption(),
      ]);
      setRules(ruleData || []);
      setAlgorithmOptions(algoData || []);
    } catch (err: unknown) {
      const messageErr = err as { message?: string };
      message.error(messageErr.message || "加载推荐规则失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleOpenModal = (record?: RecommendationRule) => {
    setEditingRule(record || null);
    if (record) {
      form.setFieldsValue(record);
    } else {
      form.resetFields();
      form.setFieldsValue({
        ruleName: "",
        sceneType: "",
        algorithmIds: [],
        weight: 50,
        enabled: true,
      });
    }
    setModalVisible(true);
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      if (editingRule?.id) {
        await RecommendationAPI.updateRule(editingRule.id, values);
        message.success("规则更新成功");
      } else {
        Modal.info({
          title: "提示",
          content:
            "规则创建功能待后端接口开放后启用，当前仅支持编辑和开关状态。",
          okText: "确定",
        });
      }
      setModalVisible(false);
      loadData();
    } catch (err: unknown) {
      const formErr = err as { errorFields?: Array<{ name: unknown }> };
      if (formErr.errorFields) return;
      const messageErr = err as { message?: string };
      message.error(messageErr.message || "保存失败");
    }
  };

  const handleDelete = useCallback((id: number) => {
    Modal.info({
      title: "提示",
      content: "规则删除功能待后端接口开放后启用。",
      okText: "确定",
    });
  }, []);

  const handleToggleEnabled = async (id: number, currentEnabled: boolean) => {
    try {
      const rule = rules.find((r) => r.id === id);
      if (!rule) return;
      await RecommendationAPI.updateRule(id, {
        ...rule,
        enabled: !currentEnabled,
      });
      message.success(currentEnabled ? "规则已禁用" : "规则已启用");
      loadData();
    } catch (err: unknown) {
      const messageErr = err as { message?: string };
      message.error(messageErr.message || "操作失败");
    }
  };

  const columns: TableColumnsType<RecommendationRule> = useMemo(
    () => [
      {
        title: "规则名称",
        dataIndex: "ruleName",
        key: "ruleName",
        width: 150,
      },
      {
        title: "场景类型",
        dataIndex: "sceneType",
        key: "sceneType",
        width: 120,
        render: (text: string) => <Tag color="blue">{text}</Tag>,
      },
      {
        title: "推荐算法",
        dataIndex: "algorithmIds",
        key: "algorithmIds",
        width: 200,
        render: (ids: number[]) => (
          <Space wrap>
            {(ids || []).map((id) => {
              const algo = algorithmOptions.find((a) => a.value === id);
              return algo ? <Tag key={id}>{algo.label}</Tag> : null;
            })}
          </Space>
        ),
      },
      {
        title: "权重",
        dataIndex: "weight",
        key: "weight",
        width: 100,
        align: "center",
        render: (weight: number, record: RecommendationRule) => (
          <InputNumber
            value={weight}
            onChange={(val) => {
              if (record.id && val !== undefined) {
                setRules((prev) =>
                  prev.map((r) =>
                    r.id === record.id ? { ...r, weight: val! } : r
                  )
                );
              }
            }}
            style={{ width: 80 }}
            min={0}
            max={100}
            step={5}
            onBlur={() => {}}
          />
        ),
      },
      {
        title: "状态",
        dataIndex: "enabled",
        key: "enabled",
        width: 80,
        align: "center",
        render: (enabled: boolean, record: RecommendationRule) => (
          <Switch
            checked={enabled}
            onChange={() =>
              record.id && handleToggleEnabled(record.id, enabled)
            }
            checkedChildren={<CheckCircleOutlined />}
            unCheckedChildren={<CloseCircleOutlined />}
          />
        ),
      },
      {
        title: "操作",
        key: "action",
        width: 160,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: RecommendationRule) => (
          <Space size="small">
            <Tooltip title="编辑">
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleOpenModal(record)}
              />
            </Tooltip>
            <Popconfirm
              title="确认删除该规则？"
              onConfirm={() => record.id && handleDelete(record.id)}
              okText="确定"
              cancelText="取消"
            >
              <Button
                type="link"
                size="small"
                danger
                icon={<DeleteOutlined />}
              />
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [algorithmOptions, handleDelete]
  );

  return (
    <div className="rec-rules-page">
      <Card>
        <div className="toolbar">
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => handleOpenModal()}
            >
              新增规则
            </Button>
            <Button icon={<ReloadOutlined />} onClick={loadData}>
              刷新
            </Button>
          </Space>
        </div>

        <Table
          columns={columns}
          dataSource={rules}
          rowKey={(record) => record.id?.toString() || Math.random().toString()}
          loading={loading}
          scroll={{ x: 900 }}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Modal
        title={editingRule ? "编辑推荐规则" : "新增推荐规则"}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={600}
        forceRender
      >
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            name="ruleName"
            label="规则名称"
            rules={[{ required: true, message: "请输入规则名称" }]}
          >
            <Input placeholder="例如：城市白天去雾" />
          </Form.Item>

          <Form.Item
            name="sceneType"
            label="场景类型"
            rules={[{ required: true, message: "请选择场景类型" }]}
          >
            <Select
              placeholder="选择场景类型"
              options={[
                { label: "城市", value: "urban" },
                { label: "风景", value: "landscape" },
                { label: "建筑", value: "building" },
                { label: "夜景", value: "night" },
                { label: "逆光", value: "backlight" },
                { label: "室内", value: "indoor" },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="algorithmIds"
            label="推荐算法"
            rules={[{ required: true, message: "请至少选择一个算法" }]}
          >
            <Select
              mode="multiple"
              placeholder="选择推荐算法"
              options={algorithmOptions}
              allowClear
            />
          </Form.Item>

          <Form.Item
            name="weight"
            label="权重 (0-100)"
            rules={[
              { required: true, message: "请输入权重" },
              { type: "number", min: 0, max: 100, message: "权重范围为 0-100" },
            ]}
          >
            <InputNumber
              style={{ width: "100%" }}
              min={0}
              max={100}
              step={5}
              suffix="%"
            />
          </Form.Item>

          <Form.Item name="enabled" label="启用" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<SaveOutlined />}>
                保存
              </Button>
              <Button onClick={() => setModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
