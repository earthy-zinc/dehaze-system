package com.pei.dehaze.ui.dashboard.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;

public class StatAdapter extends ListAdapter<StatAdapter.StatItem, StatAdapter.StatViewHolder> {

    public StatAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<StatItem> DIFF_CALLBACK = new DiffUtil.ItemCallback<StatItem>() {
        @Override
        public boolean areItemsTheSame(@NonNull StatItem oldItem, @NonNull StatItem newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull StatItem oldItem, @NonNull StatItem newItem) {
            return oldItem.getTitle().equals(newItem.getTitle()) &&
                    oldItem.getValue() == newItem.getValue() &&
                    oldItem.getDescription().equals(newItem.getDescription());
        }
    };

    @NonNull
    @Override
    public StatViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_stat, parent, false);
        return new StatViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull StatViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    static class StatViewHolder extends RecyclerView.ViewHolder {
        private TextView tvTitle;
        private TextView tvValue;
        private TextView tvDescription;

        StatViewHolder(@NonNull View itemView) {
            super(itemView);
            tvTitle = itemView.findViewById(R.id.tv_title);
            tvValue = itemView.findViewById(R.id.tv_value);
            tvDescription = itemView.findViewById(R.id.tv_description);
        }

        void bind(StatItem statItem) {
            tvTitle.setText(statItem.getTitle());
            tvValue.setText(String.valueOf(statItem.getValue()));
            tvDescription.setText(statItem.getDescription());
        }
    }

    public static class StatItem {
        private int id;
        private String title;
        private int value;
        private String description;

        public StatItem(int id, String title, int value, String description) {
            this.id = id;
            this.title = title;
            this.value = value;
            this.description = description;
        }

        public int getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public int getValue() {
            return value;
        }

        public String getDescription() {
            return description;
        }
    }
}