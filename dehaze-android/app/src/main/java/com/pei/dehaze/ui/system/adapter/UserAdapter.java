package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.user.UserPageVO;

public class UserAdapter extends ListAdapter<UserPageVO, UserAdapter.UserViewHolder> {
    
    public UserAdapter() {
        super(DIFF_CALLBACK);
    }
    
    private static final DiffUtil.ItemCallback<UserPageVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<UserPageVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull UserPageVO oldItem, @NonNull UserPageVO newItem) {
            return oldItem.getId() == newItem.getId();
        }
        
        @Override
        public boolean areContentsTheSame(@NonNull UserPageVO oldItem, @NonNull UserPageVO newItem) {
            return oldItem.getUsername().equals(newItem.getUsername()) &&
                   oldItem.getNickname().equals(newItem.getNickname()) &&
                   oldItem.getMobile().equals(newItem.getMobile());
        }
    };
    
    @NonNull
    @Override
    public UserViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_user, parent, false);
        return new UserViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(@NonNull UserViewHolder holder, int position) {
        holder.bind(getItem(position));
    }
    
    static class UserViewHolder extends RecyclerView.ViewHolder {
        private TextView tvUsername;
        private TextView tvNickname;
        private TextView tvMobile;
        
        UserViewHolder(@NonNull View itemView) {
            super(itemView);
            tvUsername = itemView.findViewById(R.id.tv_username);
            tvNickname = itemView.findViewById(R.id.tv_nickname);
            tvMobile = itemView.findViewById(R.id.tv_mobile);
        }
        
        void bind(UserPageVO user) {
            tvUsername.setText(user.getUsername());
            tvNickname.setText(user.getNickname());
            tvMobile.setText(user.getMobile());
        }
    }
}