package com.pei.dehaze.ui.algorithm_select.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

public class AlgorithmCompareResultAdapter extends ListAdapter<AlgorithmCompareVO, AlgorithmCompareResultAdapter.CompareViewHolder> {

    public AlgorithmCompareResultAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<AlgorithmCompareVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<AlgorithmCompareVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull AlgorithmCompareVO oldItem, @NonNull AlgorithmCompareVO newItem) {
            return Objects.equals(oldItem.getAlgorithmId(), newItem.getAlgorithmId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull AlgorithmCompareVO oldItem, @NonNull AlgorithmCompareVO newItem) {
            return Objects.equals(oldItem.getTime(), newItem.getTime()) &&
                    Objects.equals(oldItem.getAlgorithmName(), newItem.getAlgorithmName()) &&
                    Objects.equals(oldItem.getResultUrl(), newItem.getResultUrl()) &&
                    Objects.equals(oldItem.getMetrics(), newItem.getMetrics());
        }
    };

    @NonNull
    @Override
    public CompareViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm_compare_result, parent, false);
        return new CompareViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull CompareViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    /**
     * 解析后端返回的指标 JSON（如 {"psnr":38.2,"ssim":0.98}），输出可读字符串
     */
    private static String formatMetrics(String metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return null;
        }
        try {
            JSONObject json = new JSONObject(metrics);
            List<String> parts = new ArrayList<>();
            Iterator<String> keys = json.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                parts.add(key.toUpperCase() + " " + json.opt(key));
            }
            return parts.isEmpty() ? null : String.join(" · ", parts);
        } catch (Exception e) {
            return metrics;
        }
    }

    static class CompareViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvProcessTime;
        private TextView tvMetrics;
        private TextView tvResultUrl;

        CompareViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvProcessTime = itemView.findViewById(R.id.tv_process_time);
            tvMetrics = itemView.findViewById(R.id.tv_metrics);
            tvResultUrl = itemView.findViewById(R.id.tv_result_url);
        }

        void bind(AlgorithmCompareVO vo) {
            tvName.setText(StringUtils.safe(vo.getAlgorithmName(), "-"));
            tvProcessTime.setText("处理耗时: " + (vo.getTime() != null ? vo.getTime() + " ms" : "-"));
            tvMetrics.setText("评估指标: " + StringUtils.safe(formatMetrics(vo.getMetrics()), "-"));
            tvResultUrl.setText("结果: " + StringUtils.safe(vo.getResultUrl(), "-"));
        }
    }
}
