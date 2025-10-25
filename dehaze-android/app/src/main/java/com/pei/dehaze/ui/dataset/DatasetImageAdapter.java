package com.pei.dehaze.ui.dataset;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.ui.dataset.model.ViewCard;

public class DatasetImageAdapter extends ListAdapter<ViewCard, DatasetImageAdapter.ImageViewHolder> {

    private OnItemClickListener listener;

    public DatasetImageAdapter(OnItemClickListener listener) {
        super(DIFF_CALLBACK);
        this.listener = listener;
    }

    private static final DiffUtil.ItemCallback<ViewCard> DIFF_CALLBACK = new DiffUtil.ItemCallback<ViewCard>() {
        @Override
        public boolean areItemsTheSame(@NonNull ViewCard oldItem, @NonNull ViewCard newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull ViewCard oldItem, @NonNull ViewCard newItem) {
            return oldItem.getSrc().equals(newItem.getSrc()) &&
                    oldItem.getAlt().equals(newItem.getAlt());
        }
    };

    @NonNull
    @Override
    public ImageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dataset_image, parent, false);
        return new ImageViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ImageViewHolder holder, int position) {
        ViewCard currentItem = getItem(position);
        holder.bind(currentItem);
    }

    class ImageViewHolder extends RecyclerView.ViewHolder {
        private ImageView imageView;

        ImageViewHolder(@NonNull View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.image_view);
            
            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onItemClick(getItem(position), position);
                }
            });
        }

        void bind(ViewCard image) {
            // 使用 Glide 加载图片
            Glide.with(itemView.getContext())
                    .load(image.getSrc())
                    .placeholder(R.drawable.ic_image)
                    .error(R.drawable.ic_broken_image)
                    .into(imageView);
        }
    }

    public interface OnItemClickListener {
        void onItemClick(ViewCard image, int position);
    }
}