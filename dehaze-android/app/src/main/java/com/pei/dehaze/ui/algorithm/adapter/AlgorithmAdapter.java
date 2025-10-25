package com.pei.dehaze.ui.algorithm.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;

public class AlgorithmAdapter extends ListAdapter<Algorithm, AlgorithmAdapter.AlgorithmViewHolder> {

    private OnAlgorithmClickListener listener;

    public AlgorithmAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<Algorithm> DIFF_CALLBACK = new DiffUtil.ItemCallback<Algorithm>() {
        @Override
        public boolean areItemsTheSame(@NonNull Algorithm oldItem, @NonNull Algorithm newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull Algorithm oldItem, @NonNull Algorithm newItem) {
            return oldItem.getName().equals(newItem.getName()) &&
                    oldItem.getType().equals(newItem.getType()) &&
                    oldItem.getDescription().equals(newItem.getDescription());
        }
    };

    @NonNull
    @Override
    public AlgorithmViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm, parent, false);
        return new AlgorithmViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull AlgorithmViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    public void setOnAlgorithmClickListener(OnAlgorithmClickListener listener) {
        this.listener = listener;
    }

    class AlgorithmViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvType;
        private TextView tvDescription;

        AlgorithmViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_algorithm_name);
            tvType = itemView.findViewById(R.id.tv_algorithm_type);
            tvDescription = itemView.findViewById(R.id.tv_algorithm_description);

            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onAlgorithmClick(getItem(position));
                }
            });
        }

        void bind(Algorithm algorithm) {
            tvName.setText(algorithm.getName());
            tvType.setText(algorithm.getType());
            tvDescription.setText(algorithm.getDescription());
        }
    }

    public interface OnAlgorithmClickListener {
        void onAlgorithmClick(Algorithm algorithm);
    }
}