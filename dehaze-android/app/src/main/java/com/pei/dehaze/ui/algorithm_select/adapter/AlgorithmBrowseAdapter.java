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

import java.util.Objects;

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
                    Objects.equals(oldItem.getName(), newItem.getName()) &&
                    Objects.equals(oldItem.getType(), newItem.getType()) &&
                    Objects.equals(oldItem.getDescription(), newItem.getDescription()) &&
                    Objects.equals(oldItem.getStatus(), newItem.getStatus());
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

            AlgorithmStatus status = algorithm.getStatus() != null ? algorithm.getStatus() : AlgorithmStatus.DRAFT;
            chipStatus.setText(status.getLabel());
            chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(status)));

            btnUse.setOnClickListener(v -> {
                if (listener != null) listener.onUse(algorithm);
            });
            btnFavorite.setOnClickListener(v -> {
                if (listener != null) listener.onFavorite(algorithm);
            });
        }

        private int statusColor(AlgorithmStatus status) {
            if (status == null) return 0xFF9E9E9E;
            switch (status) {
                case DRAFT: return 0xFF9E9E9E;
                case TESTING: return 0xFFFF9800;
                case PENDING_AUDIT: return 0xFF2196F3;
                case PUBLISHED: return 0xFF4CAF50;
                case DISABLED: return 0xFFE53935;
                case ARCHIVED: return 0xFF607D8B;
                default: return 0xFF9E9E9E;
            }
        }
    }
}
