package com.pei.dehaze.ui.tools;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;

import java.util.ArrayList;
import java.util.List;

public class FeatureGridAdapter extends RecyclerView.Adapter<FeatureGridAdapter.ViewHolder> {

    private List<ToolsViewModel.FeatureItem> items = new ArrayList<>();
    private final OnFeatureClickListener listener;

    public interface OnFeatureClickListener {
        void onClick(ToolsViewModel.FeatureItem item);
    }

    public FeatureGridAdapter(OnFeatureClickListener listener) {
        this.listener = listener;
    }

    public void submitList(List<ToolsViewModel.FeatureItem> newList) {
        this.items = newList != null ? newList : new ArrayList<>();
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_feature_grid, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        ToolsViewModel.FeatureItem item = items.get(position);
        holder.tvName.setText(item.getName());
        Context ctx = holder.itemView.getContext();
        int iconRes = ctx.getResources().getIdentifier(item.getIconName(), "drawable", ctx.getPackageName());
        if (iconRes != 0) {
            holder.ivIcon.setImageResource(iconRes);
        }
        holder.itemView.setOnClickListener(v -> {
            if (listener != null) listener.onClick(item);
        });
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        final ImageView ivIcon;
        final TextView tvName;

        ViewHolder(View itemView) {
            super(itemView);
            ivIcon = itemView.findViewById(R.id.ivIcon);
            tvName = itemView.findViewById(R.id.tvName);
        }
    }
}
