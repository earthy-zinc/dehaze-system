package com.pei.dehaze.ui.messages;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.message.MessageVO;

import java.util.ArrayList;
import java.util.List;

public class MessagesAdapter extends RecyclerView.Adapter<MessagesAdapter.ViewHolder> {

    private List<MessageVO> items = new ArrayList<>();
    private final OnItemClickListener listener;

    public interface OnItemClickListener {
        void onClick(MessageVO item);
    }

    public MessagesAdapter(OnItemClickListener listener) {
        this.listener = listener;
    }

    public void submitList(List<MessageVO> newList) {
        this.items = newList != null ? newList : new ArrayList<>();
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_message, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        MessageVO item = items.get(position);
        holder.tvTitle.setText(item.getTitle());
        holder.tvSummary.setText(item.getSummary());
        // typeLabel 后端返回（如"站内信"），缺失时回退到 type 原始值
        holder.tvType.setText(item.getTypeLabel() != null ? item.getTypeLabel() : item.getType());

        // createTime 后端返回 "yyyy-MM-dd HH:mm:ss"，截取 MM-dd HH:mm 显示
        String time = item.getCreateTime();
        if (time != null && time.length() >= 16) {
            holder.tvTime.setText(time.substring(5, 16));
        } else {
            holder.tvTime.setText(time != null ? time : "");
        }

        boolean read = item.getReadStatus() != null && item.getReadStatus() == 1;
        holder.tvTitle.setAlpha(read ? 0.6f : 1f);
        holder.dotUnread.setVisibility(read ? View.INVISIBLE : View.VISIBLE);

        holder.itemView.setOnClickListener(v -> {
            if (listener != null) listener.onClick(item);
        });
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        final View dotUnread;
        final TextView tvTitle;
        final TextView tvSummary;
        final TextView tvType;
        final TextView tvTime;

        ViewHolder(View itemView) {
            super(itemView);
            dotUnread = itemView.findViewById(R.id.dotUnread);
            tvTitle = itemView.findViewById(R.id.tvTitle);
            tvSummary = itemView.findViewById(R.id.tvSummary);
            tvType = itemView.findViewById(R.id.tvType);
            tvTime = itemView.findViewById(R.id.tvTime);
        }
    }
}
