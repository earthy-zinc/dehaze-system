package com.pei.dehaze.ui.batch.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.card.MaterialCardView;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.utils.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 批量处理算法选择适配器（单选，选中高亮）
 */
public class BatchAlgorithmAdapter extends RecyclerView.Adapter<BatchAlgorithmAdapter.ViewHolder> {

    public interface OnAlgorithmSelectListener {
        void onSelect(Algorithm algorithm);
    }

    private final List<Algorithm> items = new ArrayList<>();
    private long selectedId = -1;
    private OnAlgorithmSelectListener listener;

    public void setOnAlgorithmSelectListener(OnAlgorithmSelectListener listener) {
        this.listener = listener;
    }

    public void submitList(List<Algorithm> newItems) {
        items.clear();
        if (newItems != null) {
            items.addAll(newItems);
        }
        notifyDataSetChanged();
    }

    public Algorithm getSelected() {
        for (Algorithm a : items) {
            if (a.getId() != null && a.getId() == selectedId) {
                return a;
            }
        }
        return null;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_batch_algorithm_selector, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Algorithm algo = items.get(position);
        holder.bind(algo, algo.getId() != null && algo.getId() == selectedId);
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private final MaterialCardView card;
        private final TextView tvName;
        private final TextView tvDesc;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            card = (MaterialCardView) itemView;
            tvName = itemView.findViewById(R.id.tv_algo_name);
            tvDesc = itemView.findViewById(R.id.tv_algo_desc);
        }

        void bind(Algorithm algo, boolean isSelected) {
            tvName.setText(StringUtils.safe(algo.getName()));
            tvDesc.setText(StringUtils.safe(algo.getDescription()));
            card.setStrokeWidth(isSelected ? 2 : 0);

            card.setOnClickListener(v -> {
                long id = algo.getId() != null ? algo.getId() : -1;
                if (id == selectedId) return;
                long oldId = selectedId;
                selectedId = id;
                if (oldId >= 0) {
                    int oldPos = findItemPosition(oldId);
                    if (oldPos >= 0) notifyItemChanged(oldPos);
                }
                int currentPos = getAdapterPosition();
                if (currentPos != RecyclerView.NO_POSITION) notifyItemChanged(currentPos);
                if (listener != null) listener.onSelect(algo);
            });
        }

        private int findPosition(long id) {
            for (int i = 0; i < items.size(); i++) {
                if (items.get(i).getId() != null && items.get(i).getId() == id) {
                    return i;
                }
            }
            return -1;
        }
    }

    private int findItemPosition(long id) {
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i).getId() != null && items.get(i).getId() == id) {
                return i;
            }
        }
        return -1;
    }
}
