import React from "react";
import { View, Text } from "@tarojs/components";
import { Button, Popup, Textarea } from "@taroify/core";
import type { Algorithm } from "dehaze-sdk-js";

interface AlgorithmAuditPopupProps {
  open: boolean;
  algorithm: Algorithm | null;
  approved: boolean;
  remark: string;
  submitting: boolean;
  onClose: () => void;
  onRemarkChange: (value: string) => void;
  onSubmit: () => void;
}

const AlgorithmAuditPopup: React.FC<AlgorithmAuditPopupProps> = ({
  open,
  algorithm,
  approved,
  remark,
  submitting,
  onClose,
  onRemarkChange,
  onSubmit,
}) => {
  return (
    <Popup
      open={open}
      placement="center"
      rounded
      onClose={onClose}
      className="audit-popup"
    >
      <View className="audit-content">
        <Text className="audit-title">
          {approved ? "审核通过" : "审核驳回"}
        </Text>
        {algorithm && (
          <Text className="audit-name">算法：{algorithm.name}</Text>
        )}
        {!approved && (
          <View className="audit-remark">
            <Text className="remark-label">驳回原因（必填）</Text>
            <Textarea
              className="remark-input"
              placeholder="请输入驳回原因"
              value={remark}
              onInput={(e) => onRemarkChange(e.detail.value)}
              maxlength={200}
            />
          </View>
        )}
        <View className="audit-footer">
          <Button block onClick={onClose}>
            取消
          </Button>
          <Button
            block
            color={approved ? "success" : "danger"}
            loading={submitting}
            onClick={onSubmit}
          >
            确认
          </Button>
        </View>
      </View>
    </Popup>
  );
};

export default AlgorithmAuditPopup;
