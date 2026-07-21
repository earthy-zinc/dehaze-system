package com.pei.dehaze.ui.task.adapter;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskType;
import com.pei.dehaze.sdk.model.task.TaskVO;

/**
 * 任务列表 Adapter
 */
public class TaskAdapter extends ListAdapter<TaskVO, TaskAdapter.TaskViewHolder> {

    private OnTaskClickListener listener;

    public TaskAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<TaskVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<TaskVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull TaskVO oldItem, @NonNull TaskVO newItem) {
            return oldItem.getTaskId() != null && oldItem.getTaskId().equals(newItem.getTaskId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull TaskVO oldItem, @NonNull TaskVO newItem) {
            return oldItem.getStatus().equals(newItem.getStatus()) &&
                    oldItem.getProgress() == newItem.getProgress() &&
                    (oldItem.getTaskType() == null ? newItem.getTaskType() == null : oldItem.getTaskType().equals(newItem.getTaskType()));
        }
    };

    @NonNull
    @Override
    public TaskViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_task, parent, false);
        return new TaskViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull TaskViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    public void setOnTaskClickListener(OnTaskClickListener listener) {
        this.listener = listener;
    }

    class TaskViewHolder extends RecyclerView.ViewHolder {
        private TextView tvTaskId;
        private TextView tvType;
        private TextView tvStatus;
        private TextView tvProgress;
        private TextView tvCreatedAt;
        private TextView tvCompletedAt;
        private ProgressBar progressBar;
        private TextView btnCancel;
        private TextView btnDownload;

        TaskViewHolder(@NonNull View itemView) {
            super(itemView);
            tvTaskId = itemView.findViewById(R.id.tv_task_id);
            tvType = itemView.findViewById(R.id.tv_task_type);
            tvStatus = itemView.findViewById(R.id.tv_task_status);
            tvProgress = itemView.findViewById(R.id.tv_task_progress);
            tvCreatedAt = itemView.findViewById(R.id.tv_task_created_at);
            tvCompletedAt = itemView.findViewById(R.id.tv_task_completed_at);
            progressBar = itemView.findViewById(R.id.task_progress_bar);
            btnCancel = itemView.findViewById(R.id.btn_cancel);
            btnDownload = itemView.findViewById(R.id.btn_download);

            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onTaskClick(getItem(position));
                }
            });

            btnCancel.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onCancelClick(getItem(position));
                }
            });

            btnDownload.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onDownloadClick(getItem(position));
                }
            });
        }

        void bind(TaskVO task) {
            tvTaskId.setText(task.getTaskId());
            tvType.setText(getTypeLabel(task.getTaskType()));

            TaskStatus status = TaskStatus.fromValue(task.getStatus());
            tvStatus.setText(status.getLabel());
            applyStatusColor(tvStatus, status);

            tvProgress.setText(task.getProgress() + "%");
            progressBar.setProgress(task.getProgress());
            progressBar.setVisibility(isProcessing(task.getStatus()) ? View.VISIBLE : View.GONE);

            tvCreatedAt.setText(task.getCreatedAt() != null ? task.getCreatedAt() : "");
            tvCompletedAt.setText(task.getCompletedAt() != null ? task.getCompletedAt() : "—");

            // 操作按钮显示规则
            boolean canCancel = TaskStatus.PENDING.getValue().equalsIgnoreCase(task.getStatus()) ||
                    TaskStatus.PROCESSING.getValue().equalsIgnoreCase(task.getStatus());
            boolean canDownload = TaskStatus.COMPLETED.getValue().equalsIgnoreCase(task.getStatus());
            btnCancel.setVisibility(canCancel ? View.VISIBLE : View.GONE);
            btnDownload.setVisibility(canDownload ? View.VISIBLE : View.GONE);
        }

        private String getTypeLabel(String typeValue) {
            if (typeValue == null) {
                return "未知";
            }
            for (TaskType type : TaskType.values()) {
                if (type.getValue().equals(typeValue)) {
                    return type.getLabel();
                }
            }
            return typeValue;
        }

        private void applyStatusColor(TextView tv, TaskStatus status) {
            int color;
            switch (status) {
                case PENDING:
                case PROCESSING:
                    color = Color.parseColor("#1890ff");
                    break;
                case COMPLETED:
                    color = Color.parseColor("#52c41a");
                    break;
                case FAILED:
                    color = Color.parseColor("#ff4d4f");
                    break;
                case CANCELLED:
                default:
                    color = Color.parseColor("#8c8c8c");
                    break;
            }
            tv.setTextColor(color);
        }

        private boolean isProcessing(String status) {
            return TaskStatus.PROCESSING.getValue().equalsIgnoreCase(status);
        }
    }

    public interface OnTaskClickListener {
        void onTaskClick(TaskVO task);
        void onCancelClick(TaskVO task);
        void onDownloadClick(TaskVO task);
    }
}
