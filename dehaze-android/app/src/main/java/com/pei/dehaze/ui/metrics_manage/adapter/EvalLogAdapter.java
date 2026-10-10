package com.pei.dehaze.ui.metrics_manage.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.evaluation.EvalTaskStatus;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 评估日志列表适配器（支持多选对比）
 */
public class EvalLogAdapter extends RecyclerView.Adapter<EvalLogAdapter.ViewHolder> {

    public interface OnSelectionChangeListener {
        void onSelectionChanged(Set<Long> selectedIds);
    }

    private final List<EvaluationLogVO> items = new ArrayList<>();
    private boolean compareMode = false;
    private final Set<Long> selectedIds = new HashSet<>();
    private OnSelectionChangeListener selectionListener;

    public void setSelectionListener(OnSelectionChangeListener listener) {
        this.selectionListener = listener;
    }

    public void setCompareMode(boolean compareMode) {
        this.compareMode = compareMode;
        if (!compareMode) {
            selectedIds.clear();
            notifySelectionChanged();
        }
        notifyDataSetChanged();
    }

    public boolean isCompareMode() {
        return compareMode;
    }

    public Set<Long> getSelectedIds() {
        return new HashSet<>(selectedIds);
    }

    public void clearSelection() {
        selectedIds.clear();
        notifyDataSetChanged();
        notifySelectionChanged();
    }

    public void submitList(List<EvaluationLogVO> newItems) {
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
                .inflate(R.layout.item_eval_log, parent, false);
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

    private void notifySelectionChanged() {
        if (selectionListener != null) {
            selectionListener.onSelectionChanged(new HashSet<>(selectedIds));
        }
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private final CheckBox cbSelect;
        private final TextView tvLogId;
        private final TextView tvModelId;
        private final TextView tvPsnr;
        private final TextView tvSsim;
        private final TextView tvLpips;
        private final TextView tvTime;
        private final TextView tvStatus;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            tvLogId = itemView.findViewById(R.id.tv_log_id);
            tvModelId = itemView.findViewById(R.id.tv_model_id);
            tvPsnr = itemView.findViewById(R.id.tv_psnr);
            tvSsim = itemView.findViewById(R.id.tv_ssim);
            tvLpips = itemView.findViewById(R.id.tv_lpips);
            tvTime = itemView.findViewById(R.id.tv_time);
            tvStatus = itemView.findViewById(R.id.tv_status);
        }

        void bind(EvaluationLogVO log) {
            tvLogId.setText("评估记录 #" + (log.getId() != null ? log.getId() : "--"));
            tvModelId.setText("模型: " + (log.getModelId() != null ? log.getModelId() : "--"));

            // 指标
            java.util.Map<String, Double> metrics = log.getMetrics();
            if (metrics != null) {
                tvPsnr.setText("PSNR: " + formatMetric(metrics.get("PSNR")));
                tvSsim.setText("SSIM: " + formatMetric(metrics.get("SSIM")));
                tvLpips.setText("LPIPS: " + formatMetric(metrics.get("LPIPS")));
            } else {
                tvPsnr.setText("PSNR: --");
                tvSsim.setText("SSIM: --");
                tvLpips.setText("LPIPS: --");
            }

            tvTime.setText(log.getCreateTime() != null ? log.getCreateTime() : "--");

            EvalTaskStatus status = log.getStatus();
            if (status != null) {
                tvStatus.setText(status == EvalTaskStatus.COMPLETED ? "已完成" :
                        status == EvalTaskStatus.FAILED ? "失败" : "处理中");
                tvStatus.setTextColor(status == EvalTaskStatus.COMPLETED ? 0xFF4CAF50 :
                        status == EvalTaskStatus.FAILED ? 0xFFE53935 : 0xFF2196F3);
            }

            // 对比模式
            cbSelect.setVisibility(compareMode ? View.VISIBLE : View.GONE);
            if (compareMode) {
                Long id = log.getId();
                cbSelect.setOnCheckedChangeListener(null);
                cbSelect.setChecked(id != null && selectedIds.contains(id));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (id == null) {
                        cbSelect.setChecked(false);
                        return;
                    }
                    if (checked) {
                        selectedIds.add(id);
                    } else {
                        selectedIds.remove(id);
                    }
                    notifySelectionChanged();
                });
            }
        }

        private String formatMetric(Double value) {
            if (value == null) return "--";
            return String.format("%.4f", value);
        }
    }
}
