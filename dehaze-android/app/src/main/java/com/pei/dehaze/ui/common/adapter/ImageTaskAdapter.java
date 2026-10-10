package com.pei.dehaze.ui.common.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.image_task.ImageTaskStatus;
import com.pei.dehaze.sdk.model.image_task.ImageTaskVO;

import java.util.Objects;

/**
 * 图像任务列表适配器，供 Dashboard 最近活动和 Presentation 历史画廊共用。
 */
public class ImageTaskAdapter extends ListAdapter<ImageTaskVO, ImageTaskAdapter.LogViewHolder> {

    public interface OnTaskClickListener {
        void onTaskClick(ImageTaskVO task);
    }

    private OnTaskClickListener clickListener;

    public ImageTaskAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<ImageTaskVO> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<ImageTaskVO>() {
                @Override
                public boolean areItemsTheSame(@NonNull ImageTaskVO oldItem,
                                               @NonNull ImageTaskVO newItem) {
                    return oldItem.getId() != null && oldItem.getId().equals(newItem.getId());
                }

                @Override
                public boolean areContentsTheSame(@NonNull ImageTaskVO oldItem,
                                                  @NonNull ImageTaskVO newItem) {
                    return Objects.equals(oldItem.getPredUrl(), newItem.getPredUrl())
                            && Objects.equals(oldItem.getOriginUrl(), newItem.getOriginUrl())
                            && Objects.equals(oldItem.getTime(), newItem.getTime())
                            && Objects.equals(oldItem.getStatus(), newItem.getStatus())
                            && Objects.equals(oldItem.getCreateTime(), newItem.getCreateTime());
                }
            };

    public void setOnTaskClickListener(OnTaskClickListener listener) {
        this.clickListener = listener;
    }

    @NonNull
    @Override
    public LogViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_image_task, parent, false);
        return new LogViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull LogViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class LogViewHolder extends RecyclerView.ViewHolder {
        private final ImageView ivThumb;
        private final TextView tvTitle;
        private final TextView tvStatus;
        private final TextView tvTime;
        private final TextView tvCreateTime;

        LogViewHolder(@NonNull View itemView) {
            super(itemView);
            ivThumb = itemView.findViewById(R.id.iv_thumb);
            tvTitle = itemView.findViewById(R.id.tv_title);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvTime = itemView.findViewById(R.id.tv_time);
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
        }

        void bind(ImageTaskVO task) {
            String predUrl = task.getPredUrl();
            if (predUrl != null && !predUrl.isEmpty()) {
                Glide.with(itemView.getContext())
                        .load(predUrl)
                        .placeholder(R.drawable.ic_image)
                        .error(R.drawable.ic_broken_image)
                        .centerCrop()
                        .into(ivThumb);
            } else {
                ivThumb.setImageResource(R.drawable.ic_image);
            }

            String model = task.getModel();
            tvTitle.setText("去雾任务 #" + (task.getId() == null ? "-" : task.getId())
                    + (model == null || model.isEmpty() ? "" : "（" + model + "）"));

            ImageTaskStatus status = task.getStatus();
            if (status == ImageTaskStatus.FAILED) {
                tvStatus.setText("失败");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.holo_red_dark));
            } else if (status == ImageTaskStatus.COMPLETED) {
                tvStatus.setText("完成");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.holo_green_dark));
            } else if (status == ImageTaskStatus.CANCELED) {
                tvStatus.setText("已取消");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.darker_gray));
            } else {
                tvStatus.setText("处理中");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.darker_gray));
            }

            Integer time = task.getTime();
            tvTime.setText(time == null ? "耗时：-" : ("耗时：" + time + "s"));
            tvCreateTime.setText(task.getCreateTime() == null ? "" : task.getCreateTime());

            itemView.setOnClickListener(v -> {
                if (clickListener != null) {
                    clickListener.onTaskClick(task);
                }
            });
        }
    }
}
