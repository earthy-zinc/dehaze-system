package com.pei.dehaze.ui.algorithm_select.adapter;

import android.content.res.ColorStateList;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.Chip;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

public class AlgorithmBrowseAdapter extends ListAdapter<Algorithm, AlgorithmBrowseAdapter.BrowseViewHolder> {

    public interface OnBrowseActionListener {
        void onUse(Algorithm algorithm);
        void onFavorite(Algorithm algorithm);
    }

    private OnBrowseActionListener listener;

    public AlgorithmBrowseAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<Algorithm> DIFF_CALLBACK = new DiffUtil.ItemCallback<Algorithm>() {
        @Override
        public boolean areItemsTheSame(@NonNull Algorithm oldItem, @NonNull Algorithm newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull Algorithm oldItem, @NonNull Algorithm newItem) {
            return oldItem.getId() == newItem.getId() &&
                    equals(oldItem.getName(), newItem.getName()) &&
                    equals(oldItem.getType(), newItem.getType()) &&
                    equals(oldItem.getDescription(), newItem.getDescription()) &&
                    equals(oldItem.getStatus(), newItem.getStatus());
        }

        private boolean equals(Object a, Object b) {
            return a == null ? b == null : a.equals(b);
        }
    };

    public void setOnBrowseActionListener(OnBrowseActionListener listener) {
        this.listener = listener;
    }

    @NonNull
    @Override
    public BrowseViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm_browse, parent, false);
        return new BrowseViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull BrowseViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class BrowseViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvType;
        private TextView tvDescription;
        private Chip chipStatus;
        private MaterialButton btnUse;
        private MaterialButton btnFavorite;

        BrowseViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvType = itemView.findViewById(R.id.tv_type);
            tvDescription = itemView.findViewById(R.id.tv_description);
            chipStatus = itemView.findViewById(R.id.chip_status);
            btnUse = itemView.findViewById(R.id.btn_use);
            btnFavorite = itemView.findViewById(R.id.btn_favorite);
        }

        void bind(Algorithm algorithm) {
            tvName.setText(algorithm.getName() == null ? "" : algorithm.getName());
            tvType.setText(algorithm.getType() == null ? "" : algorithm.getType());
            tvDescription.setText(algorithm.getDescription() == null ? "" : algorithm.getDescription());

            int statusValue = algorithm.getStatus() != null ? algorithm.getStatus() : 0;
            AlgorithmStatus status = AlgorithmStatus.fromValue(statusValue);
            chipStatus.setText(status.getLabel());
            chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(statusValue)));

            btnUse.setOnClickListener(v -> {
                if (listener != null) listener.onUse(algorithm);
            });
            btnFavorite.setOnClickListener(v -> {
                if (listener != null) listener.onFavorite(algorithm);
            });
        }

        private int statusColor(int status) {
            switch (status) {
                case 0: return 0xFF9E9E9E;
                case 1: return 0xFFFF9800;
                case 2: return 0xFF2196F3;
                case 3: return 0xFF4CAF50;
                case 4: return 0xFFE53935;
                case 5: return 0xFF607D8B;
                default: return 0xFF9E9E9E;
            }
        }
    }
}
