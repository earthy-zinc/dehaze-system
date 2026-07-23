package com.pei.dehaze.ui.file.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.file.FileInfo;

import java.util.Objects;

/**
 * 文件列表 Adapter
 */
public class FileAdapter extends ListAdapter<FileInfo, FileAdapter.FileViewHolder> {

    private OnFileClickListener listener;

    public FileAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<FileInfo> DIFF_CALLBACK = new DiffUtil.ItemCallback<FileInfo>() {
        @Override
        public boolean areItemsTheSame(@NonNull FileInfo oldItem, @NonNull FileInfo newItem) {
            return Objects.equals(oldItem.getId(), newItem.getId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull FileInfo oldItem, @NonNull FileInfo newItem) {
            return Objects.equals(oldItem.getName(), newItem.getName()) &&
                    Objects.equals(oldItem.getSize(), newItem.getSize()) &&
                    Objects.equals(oldItem.getType(), newItem.getType()) &&
                    Objects.equals(oldItem.getCreateTime(), newItem.getCreateTime());
        }
    };

    @NonNull
    @Override
    public FileViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_file, parent, false);
        return new FileViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull FileViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    public void setOnFileClickListener(OnFileClickListener listener) {
        this.listener = listener;
    }

    class FileViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvType;
        private TextView tvSize;
        private TextView tvCreateTime;
        private TextView btnDownload;
        private TextView btnDelete;

        FileViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_file_name);
            tvType = itemView.findViewById(R.id.tv_file_type);
            tvSize = itemView.findViewById(R.id.tv_file_size);
            tvCreateTime = itemView.findViewById(R.id.tv_file_create_time);
            btnDownload = itemView.findViewById(R.id.btn_download);
            btnDelete = itemView.findViewById(R.id.btn_delete);

            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onFileClick(getItem(position));
                }
            });

            btnDownload.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onDownloadClick(getItem(position));
                }
            });

            btnDelete.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (listener != null && position != RecyclerView.NO_POSITION) {
                    listener.onDeleteClick(getItem(position));
                }
            });
        }

        void bind(FileInfo file) {
            tvName.setText(file.getName());
            tvType.setText(file.getType() != null ? file.getType() : "未知");
            tvSize.setText(file.getSize() != null ? file.getSize() : "0");
            tvCreateTime.setText(file.getCreateTime() != null ? file.getCreateTime() : "");
        }
    }

    public interface OnFileClickListener {
        void onFileClick(FileInfo file);
        void onDownloadClick(FileInfo file);
        void onDeleteClick(FileInfo file);
    }
}
