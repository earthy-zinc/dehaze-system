package com.pei.dehaze.ui.dataset;

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
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageType;
import com.pei.dehaze.sdk.model.dataset.ImageUrl;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 数据集数据项适配器（展示数据项卡片 + 按类型展示图片 + 选择模式 + 操作按钮）
 */
public class DatasetImageAdapter extends ListAdapter<ImageItem, DatasetImageAdapter.ItemViewHolder> {

    public interface OnItemActionListener {
        void onItemClick(ImageItem item, String imageUrl);

        void onEdit(ImageItem item);

        void onDelete(ImageItem item);

        void onUploadFile(ImageItem item);

        void onDeleteFile(ImageItem item, ImageUrl url);
    }

    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Long> selectedIds);
    }

    private OnItemActionListener actionListener;
    private OnSelectionChangedListener selectionListener;

    private boolean selectionMode = false;
    private final Set<Long> selectedIds = new HashSet<>();
    /** 是否显示操作按钮（浏览版隐藏编辑/删除/上传） */
    private boolean showActions = true;

    /** 当前展示的图片类型（clear/hazy/trans） */
    private ImageType currentImageType = ImageType.HAZY;

    private static final DiffUtil.ItemCallback<ImageItem> DIFF_CALLBACK = new DiffUtil.ItemCallback<ImageItem>() {
        @Override
        public boolean areItemsTheSame(@NonNull ImageItem oldItem, @NonNull ImageItem newItem) {
            return oldItem.getId() != null && oldItem.getId().equals(newItem.getId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull ImageItem oldItem, @NonNull ImageItem newItem) {
            return Objects.equals(oldItem.getId(), newItem.getId())
                    && Objects.equals(oldItem.getName(), newItem.getName())
                    && Objects.equals(oldItem.getImageCount(), newItem.getImageCount())
                    && Objects.equals(oldItem.getHazyImages(), newItem.getHazyImages());
        }
    };

    public DatasetImageAdapter() {
        super(DIFF_CALLBACK);
    }

    public void setActionListener(OnItemActionListener listener) {
        this.actionListener = listener;
    }

    public void setSelectionListener(OnSelectionChangedListener listener) {
        this.selectionListener = listener;
    }

    public void setCurrentImageType(ImageType type) {
        this.currentImageType = type;
        notifyItemRangeChanged(0, getItemCount());
    }

    public void setShowActions(boolean showActions) {
        this.showActions = showActions;
        notifyItemRangeChanged(0, getItemCount());
    }

    public ImageType getCurrentImageType() {
        return currentImageType;
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
        List<ImageItem> list = getCurrentList();
        for (ImageItem item : list) {
            if (item.getId() != null) {
                selectedIds.add(item.getId());
            }
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

    /**
     * 根据 currentImageType 在 hazyImages 中查找匹配的图片
     */
    private ImageUrl findImageUrl(ImageItem item) {
        if (item == null) return null;
        List<ImageUrl> urls = item.getHazyImages();
        if (urls == null || urls.isEmpty()) return null;
        for (ImageUrl url : urls) {
            if (currentImageType.getValue().equalsIgnoreCase(url.getType())) {
                return url;
            }
        }
        // 回退到第一张
        return urls.get(0);
    }

    @NonNull
    @Override
    public ItemViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dataset_image, parent, false);
        return new ItemViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ItemViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class ItemViewHolder extends RecyclerView.ViewHolder {
        private final CheckBox cbSelect;
        private final ImageView imageView;
        private final TextView tvName;
        private final TextView tvImageCount;
        private final TextView tvImageType;
        private final TextView tvEdit;
        private final TextView tvDelete;
        private final TextView tvUploadFile;
        private final TextView tvDeleteFile;
        private final LinearLayout layoutActions;

        ItemViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            imageView = itemView.findViewById(R.id.image_view);
            tvName = itemView.findViewById(R.id.tv_name);
            tvImageCount = itemView.findViewById(R.id.tv_image_count);
            tvImageType = itemView.findViewById(R.id.tv_image_type);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvUploadFile = itemView.findViewById(R.id.tv_upload_file);
            tvDeleteFile = itemView.findViewById(R.id.tv_delete_file);
            layoutActions = itemView.findViewById(R.id.layout_actions);
        }

        void bind(ImageItem item) {
            tvName.setText(StringUtils.safe(item.getName()));
            Integer count = item.getImageCount();
            tvImageCount.setText(count != null ? count + " 张图片" : "0 张图片");
            tvImageType.setText("类型: " + currentImageType.getValue());

            ImageUrl imageUrl = findImageUrl(item);
            if (imageUrl != null && imageUrl.getUrl() != null && !imageUrl.getUrl().isEmpty()) {
                Glide.with(itemView.getContext())
                        .load(imageUrl.getUrl())
                        .placeholder(R.drawable.ic_image)
                        .error(R.drawable.ic_broken_image)
                        .into(imageView);
                itemView.setOnClickListener(v -> {
                    if (selectionMode) {
                        toggleSelection(item);
                    } else if (actionListener != null) {
                        actionListener.onItemClick(item, imageUrl.getUrl());
                    }
                });
                tvDeleteFile.setVisibility(View.VISIBLE);
                tvDeleteFile.setOnClickListener(v -> {
                    if (actionListener != null) {
                        actionListener.onDeleteFile(item, imageUrl);
                    }
                });
            } else {
                imageView.setImageResource(R.drawable.ic_image);
                itemView.setOnClickListener(v -> {
                    if (selectionMode) {
                        toggleSelection(item);
                    }
                });
                tvDeleteFile.setVisibility(View.GONE);
            }

            if (selectionMode) {
                cbSelect.setVisibility(View.VISIBLE);
                layoutActions.setVisibility(View.GONE);
                Long id = item.getId();
                cbSelect.setOnCheckedChangeListener(null);
                cbSelect.setChecked(id != null && selectedIds.contains(id));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (id == null) {
                        cbSelect.setChecked(false);
                        return;
                    }
                    if (checked) {
                        selectedIds.add(id);
                    } else {
                        selectedIds.remove(id);
                    }
                    notifySelectionChanged();
                });
            } else {
                cbSelect.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                layoutActions.setVisibility(showActions ? View.VISIBLE : View.GONE);
                if (showActions) {
                    tvEdit.setOnClickListener(v -> {
                        if (actionListener != null) actionListener.onEdit(item);
                    });
                    tvDelete.setOnClickListener(v -> {
                        if (actionListener != null) actionListener.onDelete(item);
                    });
                    tvUploadFile.setOnClickListener(v -> {
                        if (actionListener != null) actionListener.onUploadFile(item);
                    });
                }
            }
        }

        private void toggleSelection(ImageItem item) {
            Long id = item.getId();
            if (id == null) return;
            if (selectedIds.contains(id)) {
                selectedIds.remove(id);
            } else {
                selectedIds.add(id);
            }
            notifyItemChanged(getAdapterPosition());
            notifySelectionChanged();
        }

    }
}
