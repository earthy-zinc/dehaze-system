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

public class QuickEntryAdapter extends RecyclerView.Adapter<QuickEntryAdapter.ViewHolder> {

    private List<ToolsViewModel.QuickEntry> entries = new ArrayList<>();
    private final OnEntryClickListener listener;

    public interface OnEntryClickListener {
        void onClick(ToolsViewModel.QuickEntry entry);
    }

    public QuickEntryAdapter(OnEntryClickListener listener) {
        this.listener = listener;
    }

    public void submitList(List<ToolsViewModel.QuickEntry> newList) {
        this.entries = newList != null ? newList : new ArrayList<>();
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_quick_entry, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        ToolsViewModel.QuickEntry entry = entries.get(position);
        holder.tvName.setText(entry.getName());
        Context ctx = holder.itemView.getContext();
        int iconRes = ctx.getResources().getIdentifier(entry.getIconName(), "drawable", ctx.getPackageName());
        if (iconRes != 0) {
            holder.ivIcon.setImageResource(iconRes);
        }
        holder.itemView.setOnClickListener(v -> {
            if (listener != null) listener.onClick(entry);
        });
    }

    @Override
    public int getItemCount() {
        return entries.size();
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
