package com.pei.dehaze.ui.batch.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.ui.batch.model.BatchImageItem;

import java.util.ArrayList;
import java.util.List;

/**
 * 批量处理结果列表适配器
 */
public class BatchResultAdapter extends RecyclerView.Adapter<BatchResultAdapter.ViewHolder> {

    public interface OnResultActionListener {
        void onViewResult(BatchImageItem item);
        void onRetry(BatchImageItem item);
    }

    private final List<BatchImageItem> items = new ArrayList<>();
    private OnResultActionListener listener;

    public void setOnResultActionListener(OnResultActionListener listener) {
        this.listener = listener;
    }

    public void submitList(List<BatchImageItem> newItems) {
        items.clear();
        if (newItems != null) {
            items.addAll(newItems);
        }
        notifyDataSetChanged();
    }

    public void updateItem(BatchImageItem item) {
        int pos = findPosition(item.getIndex());
        if (pos >= 0) {
            items.set(pos, item);
            notifyItemChanged(pos);
        }
    }

    private int findPosition(int index) {
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i).getIndex() == index) {
                return i;
            }
        }
        return -1;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_batch_result, parent, false);
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
        private final ImageView ivResultThumb;
        private final TextView tvResultIndex;
        private final TextView tvResultStatus;
        private final ProgressBar progressResult;
        private final View btnViewResult;
        private final View btnRetry;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            ivResultThumb = itemView.findViewById(R.id.iv_result_thumb);
            tvResultIndex = itemView.findViewById(R.id.tv_result_index);
            tvResultStatus = itemView.findViewById(R.id.tv_result_status);
            progressResult = itemView.findViewById(R.id.progress_result);
            btnViewResult = itemView.findViewById(R.id.btn_view_result);
            btnRetry = itemView.findViewById(R.id.btn_retry);
        }

        void bind(BatchImageItem item) {
            tvResultIndex.setText("第" + (item.getIndex() + 1) + "张");

            // 缩略图
            Glide.with(itemView.getContext())
                    .load(item.getUri())
                    .placeholder(R.drawable.ic_image)
                    .into(ivResultThumb);

            switch (item.getStatus()) {
                case PENDING:
                    tvResultStatus.setText("等待处理");
                    tvResultStatus.setTextColor(0xFF9E9E9E);
                    progressResult.setVisibility(View.GONE);
                    btnViewResult.setVisibility(View.GONE);
                    btnRetry.setVisibility(View.GONE);
                    break;
                case PROCESSING:
                    tvResultStatus.setText("处理中...");
                    tvResultStatus.setTextColor(0xFF2196F3);
                    progressResult.setVisibility(View.VISIBLE);
                    progressResult.setIndeterminate(true);
                    btnViewResult.setVisibility(View.GONE);
                    btnRetry.setVisibility(View.GONE);
                    break;
                case COMPLETED:
                    tvResultStatus.setText("完成");
                    tvResultStatus.setTextColor(0xFF4CAF50);
                    progressResult.setVisibility(View.GONE);
                    btnViewResult.setVisibility(View.VISIBLE);
                    btnRetry.setVisibility(View.GONE);
                    break;
                case FAILED:
                    tvResultStatus.setText("失败: " + (item.getErrorMessage() != null ? item.getErrorMessage() : "未知错误"));
                    tvResultStatus.setTextColor(0xFFE53935);
                    progressResult.setVisibility(View.GONE);
                    btnViewResult.setVisibility(View.GONE);
                    btnRetry.setVisibility(View.VISIBLE);
                    break;
            }

            btnViewResult.setOnClickListener(v -> {
                if (listener != null) listener.onViewResult(item);
            });
            btnRetry.setOnClickListener(v -> {
                if (listener != null) listener.onRetry(item);
            });
        }
    }
}
