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
import com.pei.dehaze.sdk.model.role.RolePageVO;

public class RoleAdapter extends ListAdapter<RolePageVO, RoleAdapter.RoleViewHolder> {
    
    public RoleAdapter() {
        super(DIFF_CALLBACK);
    }
    
    private static final DiffUtil.ItemCallback<RolePageVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<RolePageVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull RolePageVO oldItem, @NonNull RolePageVO newItem) {
            return oldItem.getId() == newItem.getId();
        }
        
        @Override
        public boolean areContentsTheSame(@NonNull RolePageVO oldItem, @NonNull RolePageVO newItem) {
            return oldItem.getName().equals(newItem.getName()) &&
                   oldItem.getCode().equals(newItem.getCode());
        }
    };
    
    @NonNull
    @Override
    public RoleViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_role, parent, false);
        return new RoleViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(@NonNull RoleViewHolder holder, int position) {
        holder.bind(getItem(position));
    }
    
    static class RoleViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvCode;
        
        RoleViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvCode = itemView.findViewById(R.id.tv_code);
        }
        
        void bind(RolePageVO role) {
            tvName.setText(role.getName());
            tvCode.setText(role.getCode());
        }
    }
}