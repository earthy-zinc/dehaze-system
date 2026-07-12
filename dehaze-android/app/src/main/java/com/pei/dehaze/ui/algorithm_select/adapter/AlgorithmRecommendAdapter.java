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
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;

public class AlgorithmRecommendAdapter extends ListAdapter<AlgorithmRecommendVO, AlgorithmRecommendAdapter.RecommendViewHolder> {

    public interface OnRecommendActionListener {
        void onUse(AlgorithmRecommendVO vo);
        void onFavorite(AlgorithmRecommendVO vo);
    }

    private OnRecommendActionListener listener;

    public AlgorithmRecommendAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<AlgorithmRecommendVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<AlgorithmRecommendVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull AlgorithmRecommendVO oldItem, @NonNull AlgorithmRecommendVO newItem) {
            return oldItem.getAlgorithmId() == newItem.getAlgorithmId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull AlgorithmRecommendVO oldItem, @NonNull AlgorithmRecommendVO newItem) {
            return oldItem.getScore() == newItem.getScore() &&
                    equals(oldItem.getAlgorithmName(), newItem.getAlgorithmName()) &&
                    equals(oldItem.getReason(), newItem.getReason());
        }

        private boolean equals(Object a, Object b) {
            return a == null ? b == null : a.equals(b);
        }
    };

    public void setOnRecommendActionListener(OnRecommendActionListener listener) {
        this.listener = listener;
    }

    @NonNull
    @Override
    public RecommendViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm_recommend, parent, false);
        return new RecommendViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull RecommendViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class RecommendViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvType;
        private TextView tvReason;
        private Chip chipScore;
        private MaterialButton btnUse;
        private MaterialButton btnFavorite;

        RecommendViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvType = itemView.findViewById(R.id.tv_type);
            tvReason = itemView.findViewById(R.id.tv_reason);
            chipScore = itemView.findViewById(R.id.chip_score);
            btnUse = itemView.findViewById(R.id.btn_use);
            btnFavorite = itemView.findViewById(R.id.btn_favorite);
        }

        void bind(AlgorithmRecommendVO vo) {
            tvName.setText(vo.getAlgorithmName() == null ? "" : vo.getAlgorithmName());
            tvType.setText(vo.getType() == null ? "" : vo.getType());
            tvReason.setText(vo.getReason() == null ? "" : vo.getReason());

            long score = Math.round(vo.getScore());
            chipScore.setText("匹配度 " + score + "%");
            int color = scoreColor((int) score);
            chipScore.setChipBackgroundColor(ColorStateList.valueOf(color));

            btnUse.setOnClickListener(v -> {
                if (listener != null) listener.onUse(vo);
            });
            btnFavorite.setOnClickListener(v -> {
                if (listener != null) listener.onFavorite(vo);
            });
        }

        private int scoreColor(int score) {
            if (score >= 80) return 0xFF4CAF50;
            if (score >= 60) return 0xFFFF9800;
            if (score >= 40) return 0xFF2196F3;
            return 0xFF9E9E9E;
        }
    }
}
