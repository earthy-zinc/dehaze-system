import React from "react";
import { View, Text } from "@tarojs/components";
import { Tag, Button, Popup } from "@taroify/core";
import type { Algorithm } from "dehaze-sdk-js";
import { STATUS_INFO } from "../../utils";

interface AlgorithmDetailPopupProps {
  open: boolean;
  algorithm: Algorithm | null;
  actionLoadingId: number | null;
  canAudit: boolean;
  canEdit: boolean;
  canDelete: boolean;
  onClose: () => void;
  onToggleStatus: (algo: Algorithm) => void;
  onDelete: (algo: Algorithm) => void;
  onOpenAudit: (algo: Algorithm, approved: boolean) => void;
}

const AlgorithmDetailPopup: React.FC<AlgorithmDetailPopupProps> = ({
  open,
  algorithm,
  actionLoadingId,
  canAudit,
  canEdit,
  canDelete,
  onClose,
  onToggleStatus,
  onDelete,
  onOpenAudit,
}) => {
  /** 渲染详情项 */
  const renderDetailItem = (label: string, value: React.ReactNode) => (
    <View className="detail-item">
      <Text className="detail-label">{label}</Text>
      <View className="detail-value">{value || "-"}</View>
    </View>
  );

  return (
    <Popup
      open={open}
      placement="bottom"
      rounded
      onClose={onClose}
      className="detail-popup"
    >
      {algorithm && (
        <View className="detail-content">
          <View className="detail-header">
            <Text className="detail-title">{algorithm.name}</Text>
            <Text className="detail-close" onClick={onClose}>
              关闭
            </Text>
          </View>

          <View className="detail-section">
            <Text className="section-title">基本信息</Text>
            {renderDetailItem("算法名称", algorithm.name)}
            {renderDetailItem("算法类型", algorithm.type)}
            {renderDetailItem("描述", algorithm.description)}
            {renderDetailItem(
              "状态",
              <Tag
                color={STATUS_INFO[algorithm.status ?? 0]?.color || "default"}
                size="small"
              >
                {STATUS_INFO[algorithm.status ?? 0]?.label || "未知"}
              </Tag>
            )}
            {renderDetailItem("版本", algorithm.version)}
            {renderDetailItem("大小", algorithm.size)}
          </View>

          <View className="detail-section">
            <Text className="section-title">技术信息</Text>
            {renderDetailItem("路径", algorithm.path)}
            {renderDetailItem("导入路径", algorithm.importPath)}
            {renderDetailItem("参数", algorithm.params)}
            {renderDetailItem("计算量(FLOPs)", algorithm.flops)}
          </View>

          {(algorithm.status === 2 || algorithm.auditBy != null) && (
            <View className="detail-section">
              <Text className="section-title">审核信息</Text>
              {renderDetailItem("审核人", algorithm.auditBy)}
              {renderDetailItem("审核时间", algorithm.auditTime)}
              {renderDetailItem("审核备注", algorithm.auditRemark)}
            </View>
          )}

          {renderDetailItem("创建时间", algorithm.createTime)}

          {/* 操作按钮 */}
          <View className="detail-footer">
            {algorithm.status === 2 && canAudit && (
              <>
                <Button
                  block
                  color="success"
                  onClick={() => {
                    onClose();
                    onOpenAudit(algorithm, true);
                  }}
                >
                  审核通过
                </Button>
                <Button
                  block
                  color="danger"
                  onClick={() => {
                    onClose();
                    onOpenAudit(algorithm, false);
                  }}
                >
                  审核驳回
                </Button>
              </>
            )}
            {algorithm.status === 3 && canEdit && (
              <Button
                block
                color="warning"
                loading={actionLoadingId === algorithm.id}
                onClick={() => onToggleStatus(algorithm)}
              >
                停用算法
              </Button>
            )}
            {algorithm.status === 4 && (
              <>
                {canEdit && (
                  <Button
                    block
                    color="primary"
                    loading={actionLoadingId === algorithm.id}
                    onClick={() => onToggleStatus(algorithm)}
                  >
                    启用算法
                  </Button>
                )}
                {canDelete && (
                  <Button
                    block
                    color="danger"
                    loading={actionLoadingId === algorithm.id}
                    onClick={() => {
                      onClose();
                      onDelete(algorithm);
                    }}
                  >
                    删除算法
                  </Button>
                )}
              </>
            )}
            {algorithm.status === 0 && canDelete && (
              <Button
                block
                color="danger"
                loading={actionLoadingId === algorithm.id}
                onClick={() => {
                  onClose();
                  onDelete(algorithm);
                }}
              >
                删除算法
              </Button>
            )}
          </View>
        </View>
      )}
    </Popup>
  );
};

export default AlgorithmDetailPopup;
