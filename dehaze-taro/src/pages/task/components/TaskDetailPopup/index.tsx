import React from "react";
import { View, Text } from "@tarojs/components";
import { Popup, Tag, Button, Progress } from "@taroify/core";
import type { TaskVO } from "dehaze-sdk-js";
import { formatDateTime } from "@/utils/format";
import {
  STATUS_TAG,
  TASK_TYPE_LABEL,
  POLLING_STATUSES,
  TERMINAL_STATUSES,
} from "../../constants";

interface TaskDetailPopupProps {
  open: boolean;
  task: TaskVO | null;
  cancelLoading: boolean;
  downloadLoading: boolean;
  onClose: () => void;
  onCancel: (task: TaskVO) => void;
  onDownload: (task: TaskVO) => void;
}

/**
 * 任务详情弹窗组件
 */
const TaskDetailPopup: React.FC<TaskDetailPopupProps> = ({
  open,
  task,
  cancelLoading,
  downloadLoading,
  onClose,
  onCancel,
  onDownload,
}) => {
  /** 渲染详情弹窗中的描述项 */
  const renderDetailItem = (label: string, value: React.ReactNode) => (
    <View className="detail-item">
      <Text className="detail-label">{label}</Text>
      <View className="detail-value">{value}</View>
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
      {task && (
        <View className="detail-content">
          <View className="detail-header">
            <Text className="detail-title">任务详情</Text>
            <Text className="detail-close" onClick={onClose}>
              关闭
            </Text>
          </View>

          {renderDetailItem("任务ID", task.taskId)}
          {renderDetailItem(
            "任务类型",
            TASK_TYPE_LABEL[task.taskType || ""] || task.taskType || "-"
          )}
          {renderDetailItem(
            "状态",
            <Tag
              color={STATUS_TAG[task.status]?.color || "default"}
              size="small"
            >
              {STATUS_TAG[task.status]?.label || task.status}
            </Tag>
          )}

          {POLLING_STATUSES.includes(task.status) && (
            <View className="detail-progress">
              <Progress percent={task.progress || 0} color="primary" />
            </View>
          )}

          {task.totalFiles != null &&
            renderDetailItem(
              "文件处理",
              `${task.processedFiles || 0} / ${task.totalFiles}`
            )}
          {renderDetailItem("创建时间", formatDateTime(task.createdAt))}
          {renderDetailItem("开始时间", formatDateTime(task.startedAt))}
          {renderDetailItem("完成时间", formatDateTime(task.completedAt))}
          {task.expiresAt &&
            renderDetailItem("过期时间", formatDateTime(task.expiresAt))}
          {task.error && renderDetailItem("错误信息", task.error)}

          {/* 详情弹窗操作按钮 */}
          <View className="detail-footer">
            {POLLING_STATUSES.includes(task.status) && (
              <Button
                block
                color="danger"
                loading={cancelLoading}
                onClick={() => onCancel(task)}
              >
                取消任务
              </Button>
            )}
            {task.status === 3 && (
              <Button
                block
                color="primary"
                loading={downloadLoading}
                onClick={() => onDownload(task)}
              >
                下载结果
              </Button>
            )}
            {TERMINAL_STATUSES.includes(task.status) && task.status !== 3 && (
              <Button block onClick={onClose}>
                关闭
              </Button>
            )}
          </View>
        </View>
      )}
    </Popup>
  );
};

export default TaskDetailPopup;
