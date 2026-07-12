package com.pei.dehaze.ui.algorithm_select.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.button.MaterialButton;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;

public class AlgorithmFavoriteAdapter extends ListAdapter<FavoriteVO, AlgorithmFavoriteAdapter.FavoriteViewHolder> {

    public interface OnFavoriteActionListener {
        void onUse(FavoriteVO vo);
        void onCancelFavorite(FavoriteVO vo);
    }

    private OnFavoriteActionListener listener;

    public AlgorithmFavoriteAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<FavoriteVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<FavoriteVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull FavoriteVO oldItem, @NonNull FavoriteVO newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull FavoriteVO oldItem, @NonNull FavoriteVO newItem) {
            return equals(oldItem.getAlgorithmName(), newItem.getAlgorithmName()) &&
                    equals(oldItem.getCreateTime(), newItem.getCreateTime());
        }

        private boolean equals(Object a, Object b) {
            return a == null ? b == null : a.equals(b);
        }
    };

    public void setOnFavoriteActionListener(OnFavoriteActionListener listener) {
        this.listener = listener;
    }

    @NonNull
    @Override
    public FavoriteViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm_favorite, parent, false);
        return new FavoriteViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull FavoriteViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class FavoriteViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvTime;
        private MaterialButton btnUse;
        private MaterialButton btnCancelFavorite;

        FavoriteViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvTime = itemView.findViewById(R.id.tv_time);
            btnUse = itemView.findViewById(R.id.btn_use);
            btnCancelFavorite = itemView.findViewById(R.id.btn_cancel_favorite);
        }

        void bind(FavoriteVO vo) {
            tvName.setText(vo.getAlgorithmName() == null ? "" : vo.getAlgorithmName());
            tvTime.setText(vo.getCreateTime() == null ? "" : vo.getCreateTime());

            btnUse.setOnClickListener(v -> {
                if (listener != null) listener.onUse(vo);
            });
            btnCancelFavorite.setOnClickListener(v -> {
                if (listener != null) listener.onCancelFavorite(vo);
            });
        }
    }
}
