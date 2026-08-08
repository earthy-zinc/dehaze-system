package com.pei.dehaze.ui.dataset.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetStatistics;
import com.pei.dehaze.utils.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 数据集浏览版卡片适配器（仅展示公开/共享数据集）
 */
public class DatasetBrowseAdapter extends RecyclerView.Adapter<DatasetBrowseAdapter.ViewHolder> {

    public interface OnBrowseActionListener {
        void onView(Dataset dataset);
    }

    private final List<Dataset> items = new ArrayList<>();
    private OnBrowseActionListener listener;

    public void setOnBrowseActionListener(OnBrowseActionListener listener) {
        this.listener = listener;
    }

    public void submitList(List<Dataset> newItems) {
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
                .inflate(R.layout.item_dataset_browse_card, parent, false);
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
        private final TextView tvType;
        private final TextView tvDescription;
        private final TextView tvItemCount;
        private final TextView tvFileCount;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvType = itemView.findViewById(R.id.tv_type);
            tvDescription = itemView.findViewById(R.id.tv_description);
            tvItemCount = itemView.findViewById(R.id.tv_item_count);
            tvFileCount = itemView.findViewById(R.id.tv_file_count);

            itemView.setOnClickListener(v -> {
                int pos = getAdapterPosition();
                if (pos != RecyclerView.NO_POSITION && listener != null) {
                    listener.onView(items.get(pos));
                }
            });
        }

        void bind(Dataset dataset) {
            tvName.setText(StringUtils.safe(dataset.getName()));
            tvType.setText(StringUtils.safe(dataset.getType()));
            tvDescription.setText(StringUtils.safe(dataset.getDescription()));

            DatasetStatistics stats = dataset.getStatistics();
            if (stats != null) {
                tvItemCount.setText((stats.getItemCount() != null ? stats.getItemCount() : 0) + " 项");
                tvFileCount.setText((stats.getFileCount() != null ? stats.getFileCount() : 0) + " 文件");
            } else {
                tvItemCount.setText("0 项");
                tvFileCount.setText("0 文件");
            }
        }
    }
}
