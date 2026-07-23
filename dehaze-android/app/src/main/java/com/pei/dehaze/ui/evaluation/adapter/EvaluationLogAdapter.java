package com.pei.dehaze.ui.evaluation.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;

import java.util.Map;
import java.util.Objects;

/**
 * 评估历史列表适配器，展示 EvaluationLogVO 摘要信息。
 */
public class EvaluationLogAdapter extends ListAdapter<EvaluationLogVO, EvaluationLogAdapter.LogViewHolder> {

    public interface OnLogClickListener {
        void onLogClick(EvaluationLogVO log);
    }

    private OnLogClickListener clickListener;

    public EvaluationLogAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<EvaluationLogVO> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<EvaluationLogVO>() {
                @Override
                public boolean areItemsTheSame(@NonNull EvaluationLogVO oldItem,
                                                @NonNull EvaluationLogVO newItem) {
                    return oldItem.getId() != null && oldItem.getId().equals(newItem.getId());
                }

                @Override
                public boolean areContentsTheSame(@NonNull EvaluationLogVO oldItem,
                                                  @NonNull EvaluationLogVO newItem) {
                    return Objects.equals(oldItem.getCreateTime(), newItem.getCreateTime())
                            && Objects.equals(oldItem.getResult(), newItem.getResult());
                }
            };

    public void setOnLogClickListener(OnLogClickListener listener) {
        this.clickListener = listener;
    }

    @NonNull
    @Override
    public LogViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_evaluation_log, parent, false);
        return new LogViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull LogViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class LogViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvTitle;
        private final TextView tvQualified;
        private final TextView tvMetrics;
        private final TextView tvCreateTime;

        LogViewHolder(@NonNull View itemView) {
            super(itemView);
            tvTitle = itemView.findViewById(R.id.tv_title);
            tvQualified = itemView.findViewById(R.id.tv_qualified);
            tvMetrics = itemView.findViewById(R.id.tv_metrics);
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
        }

        void bind(EvaluationLogVO log) {
            tvTitle.setText("评估记录 #" + (log.getId() == null ? "-" : log.getId()));

            EvalResult result = log.getResult();
            if (result == null) {
                tvQualified.setText("未评估");
                tvQualified.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.darker_gray));
                tvMetrics.setText("暂无指标数据");
            } else {
                Boolean qualified = result.getQualified();
                if (qualified != null) {
                    if (qualified) {
                        tvQualified.setText("合格");
                        tvQualified.setTextColor(itemView.getContext().getResources()
                                .getColor(android.R.color.holo_green_dark));
                    } else {
                        tvQualified.setText("不合格");
                        tvQualified.setTextColor(itemView.getContext().getResources()
                                .getColor(android.R.color.holo_red_dark));
                    }
                } else {
                    tvQualified.setText("已评估");
                    tvQualified.setTextColor(itemView.getContext().getResources()
                            .getColor(android.R.color.darker_gray));
                }
                tvMetrics.setText(formatMetrics(result.getMetrics()));
            }

            tvCreateTime.setText(log.getCreateTime() == null ? "" : log.getCreateTime());

            itemView.setOnClickListener(v -> {
                if (clickListener != null) {
                    clickListener.onLogClick(log);
                }
            });
        }

        private String formatMetrics(Map<String, Double> metrics) {
            if (metrics == null || metrics.isEmpty()) {
                return "暂无指标数据";
            }
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, Double> entry : metrics.entrySet()) {
                if (sb.length() > 0) sb.append("，");
                sb.append(entry.getKey()).append("=").append(entry.getValue());
            }
            return sb.toString();
        }
    }
}
