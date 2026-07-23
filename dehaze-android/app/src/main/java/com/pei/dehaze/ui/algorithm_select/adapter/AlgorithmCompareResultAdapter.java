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
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

import java.util.Objects;

public class AlgorithmCompareResultAdapter extends ListAdapter<AlgorithmCompareVO, AlgorithmCompareResultAdapter.CompareViewHolder> {

    public AlgorithmCompareResultAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<AlgorithmCompareVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<AlgorithmCompareVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull AlgorithmCompareVO oldItem, @NonNull AlgorithmCompareVO newItem) {
            return oldItem.getAlgorithmId() == newItem.getAlgorithmId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull AlgorithmCompareVO oldItem, @NonNull AlgorithmCompareVO newItem) {
            return oldItem.getProcessTime() == newItem.getProcessTime() &&
                    Objects.equals(oldItem.getAlgorithmName(), newItem.getAlgorithmName()) &&
                    Objects.equals(oldItem.getResultUrl(), newItem.getResultUrl());
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

    static class CompareViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvType;
        private TextView tvParams;
        private TextView tvFlops;
        private TextView tvProcessTime;
        private TextView tvStatus;
        private TextView tvDescription;
        private TextView tvResultUrl;

        CompareViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvType = itemView.findViewById(R.id.tv_type);
            tvParams = itemView.findViewById(R.id.tv_params);
            tvFlops = itemView.findViewById(R.id.tv_flops);
            tvProcessTime = itemView.findViewById(R.id.tv_process_time);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvDescription = itemView.findViewById(R.id.tv_description);
            tvResultUrl = itemView.findViewById(R.id.tv_result_url);
        }

        void bind(AlgorithmCompareVO vo) {
            tvName.setText(StringUtils.safe(vo.getAlgorithmName(), "-"));
            tvType.setText("类型: " + StringUtils.safe(vo.getType(), "-"));
            tvParams.setText("参数量: " + StringUtils.safe(vo.getParams(), "-"));
            tvFlops.setText("FLOPs: " + StringUtils.safe(vo.getFlops(), "-"));
            tvProcessTime.setText("处理耗时: " + (vo.getProcessTime() != null ? vo.getProcessTime() + " ms" : "-"));
            AlgorithmStatus status = vo.getStatus();
            tvStatus.setText("状态: " + (status != null ? status.getLabel() : ""));
            tvDescription.setText(StringUtils.safe(vo.getDescription(), "-"));
            tvResultUrl.setText("结果: " + StringUtils.safe(vo.getResultUrl(), "-"));
        }

    }
}
