package com.pei.dehaze.ui.input.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 图像输入历史适配器
 */
public class InputHistoryAdapter extends ListAdapter<InputHistoryVO, InputHistoryAdapter.HistoryViewHolder> {

    public interface OnHistoryActionListener {
        void onItemClick(InputHistoryVO item);

        void onEdit(InputHistoryVO item);

        void onDelete(InputHistoryVO item);

        void onToggleFavorite(InputHistoryVO item);
    }

    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Long> selectedIds);
    }

    private OnHistoryActionListener actionListener;
    private OnSelectionChangedListener selectionListener;

    private boolean selectionMode = false;
    private final Set<Long> selectedIds = new HashSet<>();

    private static final DiffUtil.ItemCallback<InputHistoryVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<InputHistoryVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull InputHistoryVO oldItem, @NonNull InputHistoryVO newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull InputHistoryVO oldItem, @NonNull InputHistoryVO newItem) {
            return oldItem.getId() == newItem.getId()
                    && equals(oldItem.getOriginalImageUrl(), newItem.getOriginalImageUrl())
                    && equals(oldItem.getResultImageUrl(), newItem.getResultImageUrl())
                    && equals(oldItem.getStatus(), newItem.getStatus())
                    && equals(oldItem.getIsFavorite(), newItem.getIsFavorite())
                    && equals(oldItem.getAlgorithmName(), newItem.getAlgorithmName());
        }

        private boolean equals(Object a, Object b) {
            return a == null ? b == null : a.equals(b);
        }
    };

    public InputHistoryAdapter() {
        super(DIFF_CALLBACK);
    }

    public void setActionListener(OnHistoryActionListener listener) {
        this.actionListener = listener;
    }

    public void setSelectionListener(OnSelectionChangedListener listener) {
        this.selectionListener = listener;
    }

    public void setSelectionMode(boolean selectionMode) {
        this.selectionMode = selectionMode;
        if (!selectionMode) {
            selectedIds.clear();
            notifySelectionChanged();
        }
        notifyItemRangeChanged(0, getItemCount());
    }

    public boolean isSelectionMode() {
        return selectionMode;
    }

    public void selectAll() {
        List<InputHistoryVO> list = getCurrentList();
        for (InputHistoryVO item : list) {
            selectedIds.add(item.getId());
        }
        notifyItemRangeChanged(0, getItemCount());
        notifySelectionChanged();
    }

    public void clearSelection() {
        selectedIds.clear();
        notifyItemRangeChanged(0, getItemCount());
        notifySelectionChanged();
    }

    public Set<Long> getSelectedIds() {
        return new HashSet<>(selectedIds);
    }

    private void notifySelectionChanged() {
        if (selectionListener != null) {
            selectionListener.onSelectionChanged(new HashSet<>(selectedIds));
        }
    }

    @NonNull
    @Override
    public HistoryViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_input_history, parent, false);
        return new HistoryViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull HistoryViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class HistoryViewHolder extends RecyclerView.ViewHolder {
        private final CheckBox cbSelect;
        private final ImageView ivOriginal;
        private final ImageView ivResult;
        private final TextView tvAlgorithmName;
        private final TextView tvStatus;
        private final TextView tvSource;
        private final TextView tvProcessingTime;
        private final TextView tvFavorite;
        private final TextView tvSyncStatus;
        private final TextView tvCreateTime;
        private final TextView tvEdit;
        private final TextView tvDelete;
        private final TextView tvFavoriteAction;
        private final LinearLayout layoutActions;

        HistoryViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            ivOriginal = itemView.findViewById(R.id.iv_original);
            ivResult = itemView.findViewById(R.id.iv_result);
            tvAlgorithmName = itemView.findViewById(R.id.tv_algorithm_name);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvSource = itemView.findViewById(R.id.tv_source);
            tvProcessingTime = itemView.findViewById(R.id.tv_processing_time);
            tvFavorite = itemView.findViewById(R.id.tv_favorite);
            tvSyncStatus = itemView.findViewById(R.id.tv_sync_status);
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvFavoriteAction = itemView.findViewById(R.id.tv_favorite_action);
            layoutActions = itemView.findViewById(R.id.layout_actions);
        }

        void bind(InputHistoryVO item) {
            tvAlgorithmName.setText("算法: " + safe(item.getAlgorithmName()));
            tvStatus.setText("状态: " + formatStatus(item.getStatus()));
            tvSource.setText("来源: " + safe(item.getInputSource()));
            tvProcessingTime.setText(item.getProcessingTime() != null
                    ? "耗时: " + item.getProcessingTime() + "ms" : "耗时: -");
            tvFavorite.setText(item.getIsFavorite() != null && item.getIsFavorite() == 1 ? "已收藏" : "未收藏");
            tvSyncStatus.setText(item.getSyncStatus() != null && item.getSyncStatus() == 1 ? "已同步" : "未同步");
            tvCreateTime.setText(safe(item.getCreateTime()));

            loadImage(ivOriginal, item.getOriginalImageUrl());
            loadImage(ivResult, item.getResultImageUrl());

            ivOriginal.setOnClickListener(v -> {
                if (selectionMode) {
                    toggleSelection(item);
                } else if (actionListener != null) {
                    actionListener.onItemClick(item);
                }
            });
            ivResult.setOnClickListener(v -> {
                if (selectionMode) {
                    toggleSelection(item);
                } else if (actionListener != null) {
                    actionListener.onItemClick(item);
                }
            });

            if (selectionMode) {
                cbSelect.setVisibility(View.VISIBLE);
                layoutActions.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                cbSelect.setChecked(selectedIds.contains(item.getId()));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (checked) {
                        selectedIds.add(item.getId());
                    } else {
                        selectedIds.remove(item.getId());
                    }
                    notifySelectionChanged();
                });
                itemView.setOnClickListener(v -> toggleSelection(item));
            } else {
                cbSelect.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                layoutActions.setVisibility(View.VISIBLE);
                itemView.setOnClickListener(null);
                tvEdit.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onEdit(item);
                });
                tvDelete.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onDelete(item);
                });
                tvFavoriteAction.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onToggleFavorite(item);
                });
            }

            itemView.setOnLongClickListener(v -> {
                if (!selectionMode) {
                    setSelectionMode(true);
                    selectedIds.add(item.getId());
                    notifyItemRangeChanged(0, getItemCount());
                    notifySelectionChanged();
                    return true;
                }
                return false;
            });
        }

        private void toggleSelection(InputHistoryVO item) {
            if (selectedIds.contains(item.getId())) {
                selectedIds.remove(item.getId());
            } else {
                selectedIds.add(item.getId());
            }
            notifyItemChanged(getAdapterPosition());
            notifySelectionChanged();
        }

        private void loadImage(ImageView imageView, String url) {
            if (url == null || url.isEmpty()) {
                imageView.setImageResource(R.drawable.ic_image);
                return;
            }
            Glide.with(itemView.getContext())
                    .load(DehazeSDK.getInstance().resolveUrl(url))
                    .placeholder(R.drawable.ic_image)
                    .error(R.drawable.ic_broken_image)
                    .into(imageView);
        }

        private String formatStatus(Integer status) {
            if (status == null) return "未知";
            switch (status) {
                case 1:
                    return "成功";
                case 2:
                    return "失败";
                case 3:
                    return "处理中";
                default:
                    return "未知";
            }
        }

        private String safe(String s) {
            return s == null ? "" : s;
        }
    }
}
