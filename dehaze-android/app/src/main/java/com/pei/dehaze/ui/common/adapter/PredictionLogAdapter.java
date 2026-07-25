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
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.prediction.PredEvalTaskStatus;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.util.Objects;

/**
 * 预测日志列表适配器，供 Dashboard 最近活动和 Presentation 历史画廊共用。
 */
public class PredictionLogAdapter extends ListAdapter<PredictionLogVO, PredictionLogAdapter.LogViewHolder> {

    public interface OnLogClickListener {
        void onLogClick(PredictionLogVO log);
    }

    private OnLogClickListener clickListener;

    public PredictionLogAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<PredictionLogVO> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<PredictionLogVO>() {
                @Override
                public boolean areItemsTheSame(@NonNull PredictionLogVO oldItem,
                                                @NonNull PredictionLogVO newItem) {
                    return oldItem.getId() != null && oldItem.getId().equals(newItem.getId());
                }

                @Override
                public boolean areContentsTheSame(@NonNull PredictionLogVO oldItem,
                                                  @NonNull PredictionLogVO newItem) {
                    return Objects.equals(oldItem.getPredUrl(), newItem.getPredUrl())
                            && Objects.equals(oldItem.getOriginUrl(), newItem.getOriginUrl())
                            && Objects.equals(oldItem.getTime(), newItem.getTime())
                            && Objects.equals(oldItem.getStatus(), newItem.getStatus())
                            && Objects.equals(oldItem.getCreateTime(), newItem.getCreateTime());
                }
            };

    public void setOnLogClickListener(OnLogClickListener listener) {
        this.clickListener = listener;
    }

    @NonNull
    @Override
    public LogViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_prediction_log, parent, false);
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

        void bind(PredictionLogVO log) {
            String predUrl = log.getPredUrl();
            if (predUrl != null && !predUrl.isEmpty()) {
                Glide.with(itemView.getContext())
                        .load(DehazeSDK.getInstance().resolveUrl(predUrl))
                        .placeholder(R.drawable.ic_image)
                        .error(R.drawable.ic_broken_image)
                        .centerCrop()
                        .into(ivThumb);
            } else {
                ivThumb.setImageResource(R.drawable.ic_image);
            }

            tvTitle.setText("去雾记录 #" + (log.getId() == null ? "-" : log.getId()));

            PredEvalTaskStatus status = PredEvalTaskStatus.fromValue(log.getStatus());
            if (status == PredEvalTaskStatus.FAILED) {
                tvStatus.setText("失败");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.holo_red_dark));
            } else if (status == PredEvalTaskStatus.COMPLETED) {
                tvStatus.setText("完成");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.holo_green_dark));
            } else {
                tvStatus.setText("处理中");
                tvStatus.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.darker_gray));
            }

            Integer time = log.getTime();
            tvTime.setText(time == null ? "耗时：-" : ("耗时：" + time + "ms"));
            tvCreateTime.setText(log.getCreateTime() == null ? "" : log.getCreateTime());

            itemView.setOnClickListener(v -> {
                if (clickListener != null) {
                    clickListener.onLogClick(log);
                }
            });
        }
    }
}
