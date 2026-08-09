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
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.ProcessStatus;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 图像输入历史适配器
 */
public class InputHistoryAdapter extends ListAdapter<InputHistoryVO, InputHistoryAdapter.HistoryViewHolder> {

    public interface OnHistoryActionListener {
        void onItemClick(InputHistoryVO item);

        void onDelete(InputHistoryVO item);
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
                    && Objects.equals(oldItem.getOriginalImageUrl(), newItem.getOriginalImageUrl())
                    && Objects.equals(oldItem.getResultImageUrl(), newItem.getResultImageUrl())
                    && Objects.equals(oldItem.getStatus(), newItem.getStatus())
                    && Objects.equals(oldItem.getAlgorithmName(), newItem.getAlgorithmName());
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
        private final TextView tvCreateTime;
        private final TextView tvDelete;
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
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            layoutActions = itemView.findViewById(R.id.layout_actions);
        }

        void bind(InputHistoryVO item) {
            tvAlgorithmName.setText("算法: " + StringUtils.safe(item.getAlgorithmName()));
            tvStatus.setText("状态: " + formatStatus(item.getStatus()));
            tvSource.setText("来源: " + (item.getInputSource() != null ? item.getInputSource().getLabel() : ""));
            tvProcessingTime.setText(item.getProcessingTime() != null
                    ? "耗时: " + item.getProcessingTime() + "ms" : "耗时: -");
            tvCreateTime.setText(StringUtils.safe(item.getCreateTime()));

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
                tvDelete.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onDelete(item);
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
                    .load(url)
                    .placeholder(R.drawable.ic_image)
                    .error(R.drawable.ic_broken_image)
                    .into(imageView);
        }

        private String formatStatus(ProcessStatus status) {
            return status != null ? status.getLabel() : "未知";
        }

    }
}
