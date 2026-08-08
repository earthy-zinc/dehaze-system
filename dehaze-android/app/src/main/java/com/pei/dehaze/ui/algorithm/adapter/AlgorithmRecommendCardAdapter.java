package com.pei.dehaze.ui.algorithm.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;
import com.pei.dehaze.utils.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 推荐算法横向卡片适配器
 */
public class AlgorithmRecommendCardAdapter extends RecyclerView.Adapter<AlgorithmRecommendCardAdapter.ViewHolder> {

    public interface OnRecommendActionListener {
        void onUse(RecommendedAlgorithmVO vo);
    }

    private final List<RecommendedAlgorithmVO> items = new ArrayList<>();
    private OnRecommendActionListener listener;

    public void setOnRecommendActionListener(OnRecommendActionListener listener) {
        this.listener = listener;
    }

    public void submitList(List<RecommendedAlgorithmVO> newItems) {
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
                .inflate(R.layout.item_algorithm_recommend_card, parent, false);
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
        private final TextView tvMatchScore;
        private final TextView tvName;
        private final TextView tvReason;
        private final View btnUse;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvMatchScore = itemView.findViewById(R.id.tv_match_score);
            tvName = itemView.findViewById(R.id.tv_name);
            tvReason = itemView.findViewById(R.id.tv_reason);
            btnUse = itemView.findViewById(R.id.btn_use);
        }

        void bind(RecommendedAlgorithmVO vo) {
            tvMatchScore.setText((vo.getMatchScore() != null ? vo.getMatchScore() : "--") + "%");
            tvName.setText(StringUtils.safe(vo.getAlgorithmName()));
            tvReason.setText(StringUtils.safe(vo.getReason()));

            btnUse.setOnClickListener(v -> {
                if (listener != null) listener.onUse(vo);
            });
        }
    }
}
