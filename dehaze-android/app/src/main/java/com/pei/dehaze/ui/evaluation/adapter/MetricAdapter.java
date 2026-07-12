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

import java.util.Map;

/**
 * 评估指标适配器，展示 EvalResult.metrics 中的各项指标
 */
public class MetricAdapter extends ListAdapter<Map.Entry<String, Double>, MetricAdapter.MetricViewHolder> {

    public MetricAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<Map.Entry<String, Double>> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<Map.Entry<String, Double>>() {
                @Override
                public boolean areItemsTheSame(@NonNull Map.Entry<String, Double> oldItem,
                                                @NonNull Map.Entry<String, Double> newItem) {
                    return oldItem.getKey().equals(newItem.getKey());
                }

                @Override
                public boolean areContentsTheSame(@NonNull Map.Entry<String, Double> oldItem,
                                                  @NonNull Map.Entry<String, Double> newItem) {
                    return oldItem.getKey().equals(newItem.getKey())
                            && Double.compare(oldItem.getValue(), newItem.getValue()) == 0;
                }
            };

    @NonNull
    @Override
    public MetricViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_metric, parent, false);
        return new MetricViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MetricViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    static class MetricViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvLabel;
        private final TextView tvValue;
        private final TextView tvDescription;
        private final TextView tvTrend;

        MetricViewHolder(@NonNull View itemView) {
            super(itemView);
            tvLabel = itemView.findViewById(R.id.tv_label);
            tvValue = itemView.findViewById(R.id.tv_value);
            tvDescription = itemView.findViewById(R.id.tv_description);
            tvTrend = itemView.findViewById(R.id.tv_trend);
        }

        void bind(Map.Entry<String, Double> metric) {
            String name = metric.getKey();
            tvLabel.setText(name);
            tvValue.setText(String.valueOf(metric.getValue()));

            // 根据指标名称判定方向：psnr/ssim 越高越好，lpips/niqe 越低越好
            String direction = describeMetric(name);
            tvDescription.setText(direction);
            if ("higher".equals(direction)) {
                tvTrend.setText("↑");
                tvTrend.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.holo_green_dark));
            } else if ("lower".equals(direction)) {
                tvTrend.setText("↓");
                tvTrend.setTextColor(itemView.getContext().getResources()
                        .getColor(android.R.color.holo_red_dark));
            } else {
                tvTrend.setText("");
            }
        }

        private static String describeMetric(String name) {
            if (name == null) {
                return "";
            }
            switch (name.toLowerCase()) {
                case "psnr":
                case "ssim":
                    return "higher";
                case "lpips":
                case "niqe":
                    return "lower";
                default:
                    return "";
            }
        }
    }
}
