package com.pei.dehaze.ui.algorithm.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmSelectNodeVO;
import com.pei.dehaze.utils.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 算法库浏览版卡片适配器（仅展示已发布算法，含「详情」「使用该算法」操作）
 */
public class AlgorithmBrowseAdapter extends RecyclerView.Adapter<AlgorithmBrowseAdapter.ViewHolder> {

    public interface OnBrowseActionListener {
        void onViewDetail(AlgorithmSelectNodeVO algorithm);
        void onUse(AlgorithmSelectNodeVO algorithm);
    }

    private final List<AlgorithmSelectNodeVO> items = new ArrayList<>();
    private OnBrowseActionListener listener;

    public void setOnBrowseActionListener(OnBrowseActionListener listener) {
        this.listener = listener;
    }

    public void submitList(List<AlgorithmSelectNodeVO> newItems) {
        items.clear();
        if (newItems != null) {
            items.addAll(newItems);
        }
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm_browse_card, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        holder.bind(items.get(position));
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvName;
        private final TextView tvDescription;
        private final TextView tvMetricPsnr;
        private final TextView tvMetricSsim;
        private final TextView tvRecommendBadge;
        private final View layoutMetrics;
        private final View btnViewDetail;
        private final View btnUse;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvDescription = itemView.findViewById(R.id.tv_description);
            tvMetricPsnr = itemView.findViewById(R.id.tv_metric_psnr);
            tvMetricSsim = itemView.findViewById(R.id.tv_metric_ssim);
            tvRecommendBadge = itemView.findViewById(R.id.tv_recommend_badge);
            layoutMetrics = itemView.findViewById(R.id.layout_metrics);
            btnViewDetail = itemView.findViewById(R.id.btn_view_detail);
            btnUse = itemView.findViewById(R.id.btn_use);
        }

        void bind(AlgorithmSelectNodeVO algorithm) {
            tvName.setText(StringUtils.safe(algorithm.getName()));
            tvDescription.setText(StringUtils.safe(algorithm.getDescription()));
            tvRecommendBadge.setVisibility(algorithm.getIsRecommended() != null && algorithm.getIsRecommended() ? View.VISIBLE : View.GONE);

            // 指标展示
            if (algorithm.getAvgPsnr() != null || algorithm.getAvgSsim() != null) {
                layoutMetrics.setVisibility(View.VISIBLE);
                tvMetricPsnr.setText("PSNR: " + (algorithm.getAvgPsnr() != null ? String.format("%.2f", algorithm.getAvgPsnr()) : "--"));
                tvMetricSsim.setText("SSIM: " + (algorithm.getAvgSsim() != null ? String.format("%.4f", algorithm.getAvgSsim()) : "--"));
            } else {
                layoutMetrics.setVisibility(View.GONE);
            }

            btnViewDetail.setOnClickListener(v -> {
                if (listener != null) listener.onViewDetail(algorithm);
            });

            btnUse.setOnClickListener(v -> {
                if (listener != null) listener.onUse(algorithm);
            });
        }
    }
}
