import React from "react";
import { View, Text } from "@tarojs/components";
import { Tag, Button, Progress } from "@taroify/core";
import type { TaskVO } from "dehaze-sdk-js";
import { formatDateTime } from "@/utils/format";
import {
  STATUS_TAG,
  TASK_TYPE_LABEL,
  POLLING_STATUSES,
  shortTaskId,
} from "../../constants";

interface TaskCardProps {
  task: TaskVO;
  cancelLoading: boolean;
  downloadLoading: boolean;
  onClick: (task: TaskVO) => void;
  onCancel: (task: TaskVO) => void;
  onDownload: (task: TaskVO) => void;
}

/**
 * 任务卡片组件
 */
const TaskCard: React.FC<TaskCardProps> = ({
  task,
  cancelLoading,
  downloadLoading,
  onClick,
  onCancel,
  onDownload,
}) => {
  const tagInfo = STATUS_TAG[task.status] || {
    label: task.status,
    color: "#8c8c8c",
  };
  const isActive = POLLING_STATUSES.includes(task.status);
  const canDownload = task.status === "COMPLETED";

  return (
    <View key={task.taskId} className="task-card" onClick={() => onClick(task)}>
      <View className="card-header">
        <View className="header-left">
          <Tag color={tagInfo.color} size="small">
            {tagInfo.label}
          </Tag>
          <Text className="task-type">
            {TASK_TYPE_LABEL[task.taskType || ""] || task.taskType || "未知"}
          </Text>
        </View>
        <Text className="task-id">{shortTaskId(task.taskId)}</Text>
      </View>

      {isActive && (
        <View className="card-progress">
          <Progress percent={task.progress || 0} color="primary" />
        </View>
      )}

      {task.status === "FAILED" && task.error && (
        <View className="card-error">
          <Text>{task.error}</Text>
        </View>
      )}

      <View className="card-footer">
        <Text className="task-time">
          创建: {formatDateTime(task.createdAt)}
        </Text>
        <View className="task-actions" onClick={(e) => e.stopPropagation()}>
          {isActive && (
            <Button
              size="mini"
              color="danger"
              loading={cancelLoading}
              onClick={() => onCancel(task)}
            >
              取消
            </Button>
          )}
          {canDownload && (
            <Button
              size="mini"
              color="primary"
              loading={downloadLoading}
              onClick={() => onDownload(task)}
            >
              下载
            </Button>
          )}
        </View>
      </View>
    </View>
  );
};

export default TaskCard;
