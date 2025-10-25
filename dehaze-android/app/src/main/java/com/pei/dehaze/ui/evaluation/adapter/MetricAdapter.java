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
import com.pei.dehaze.sdk.model.model.EvalResult;

public class MetricAdapter extends ListAdapter<EvalResult, MetricAdapter.MetricViewHolder> {

    public MetricAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<EvalResult> DIFF_CALLBACK = new DiffUtil.ItemCallback<EvalResult>() {
        @Override
        public boolean areItemsTheSame(@NonNull EvalResult oldItem, @NonNull EvalResult newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull EvalResult oldItem, @NonNull EvalResult newItem) {
            return oldItem.getLabel().equals(newItem.getLabel()) &&
                    oldItem.getValue().equals(newItem.getValue()) &&
                    oldItem.getDescription().equals(newItem.getDescription());
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
        private TextView tvLabel;
        private TextView tvValue;
        private TextView tvDescription;
        private TextView tvTrend;

        MetricViewHolder(@NonNull View itemView) {
            super(itemView);
            tvLabel = itemView.findViewById(R.id.tv_label);
            tvValue = itemView.findViewById(R.id.tv_value);
            tvDescription = itemView.findViewById(R.id.tv_description);
            tvTrend = itemView.findViewById(R.id.tv_trend);
        }

        void bind(EvalResult evalResult) {
            tvLabel.setText(evalResult.getLabel());
            tvValue.setText(evalResult.getValue());
            tvDescription.setText(evalResult.getDescription());

            // 根据 better 字段设置趋势指示器
            if ("higher".equals(evalResult.getBetter())) {
                tvTrend.setText("↑");
                tvTrend.setTextColor(itemView.getContext().getResources().getColor(android.R.color.holo_green_dark));
            } else if ("lower".equals(evalResult.getBetter())) {
                tvTrend.setText("↓");
                tvTrend.setTextColor(itemView.getContext().getResources().getColor(android.R.color.holo_red_dark));
            } else {
                tvTrend.setText("");
            }
        }
    }
}