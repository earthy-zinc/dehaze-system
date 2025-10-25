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
import com.pei.dehaze.sdk.model.dept.DeptVO;

public class DeptAdapter extends ListAdapter<DeptVO, DeptAdapter.DeptViewHolder> {
    
    public DeptAdapter() {
        super(DIFF_CALLBACK);
    }
    
    private static final DiffUtil.ItemCallback<DeptVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<DeptVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull DeptVO oldItem, @NonNull DeptVO newItem) {
            return oldItem.getId() == newItem.getId();
        }
        
        @Override
        public boolean areContentsTheSame(@NonNull DeptVO oldItem, @NonNull DeptVO newItem) {
            return oldItem.getName().equals(newItem.getName()) &&
                   oldItem.getStatus() == newItem.getStatus();
        }
    };
    
    @NonNull
    @Override
    public DeptViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dept, parent, false);
        return new DeptViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(@NonNull DeptViewHolder holder, int position) {
        holder.bind(getItem(position));
    }
    
    static class DeptViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvStatus;
        
        DeptViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvStatus = itemView.findViewById(R.id.tv_status);
        }
        
        void bind(DeptVO dept) {
            tvName.setText(dept.getName());
            tvStatus.setText(dept.getStatus() == 1 ? "正常" : "禁用");
        }
    }
}