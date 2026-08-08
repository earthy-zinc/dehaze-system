package com.pei.dehaze.ui.batch.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.ui.batch.model.BatchImageItem;

import java.util.ArrayList;
import java.util.List;

/**
 * 批量图片缩略图网格适配器
 */
public class BatchImageAdapter extends RecyclerView.Adapter<BatchImageAdapter.ViewHolder> {

    public interface OnRemoveListener {
        void onRemove(int position);
    }

    private final List<BatchImageItem> items = new ArrayList<>();
    private OnRemoveListener listener;

    public void setOnRemoveListener(OnRemoveListener listener) {
        this.listener = listener;
    }

    public void addItem(BatchImageItem item) {
        items.add(item);
        notifyItemInserted(items.size() - 1);
    }

    public void removeItem(int position) {
        if (position >= 0 && position < items.size()) {
            items.remove(position);
            notifyItemRemoved(position);
            notifyItemRangeChanged(position, items.size() - position);
        }
    }

    public List<BatchImageItem> getItems() {
        return new ArrayList<>(items);
    }

    public void clear() {
        items.clear();
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_batch_image, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        BatchImageItem item = items.get(position);
        holder.ivThumbnail.setImageURI(item.getUri());
        holder.ivRemove.setOnClickListener(v -> {
            int pos = holder.getAdapterPosition();
            if (listener != null && pos != RecyclerView.NO_POSITION) listener.onRemove(pos);
        });
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        final ImageView ivThumbnail;
        final ImageView ivRemove;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            ivThumbnail = itemView.findViewById(R.id.iv_thumbnail);
            ivRemove = itemView.findViewById(R.id.iv_remove);
        }
    }
}
