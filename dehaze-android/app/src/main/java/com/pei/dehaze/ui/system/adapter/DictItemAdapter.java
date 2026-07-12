package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.dict.DictPageVO;

public class DictItemAdapter extends ListAdapter<DictPageVO, DictItemAdapter.DictItemViewHolder> {

    public DictItemAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<DictPageVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<DictPageVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull DictPageVO oldItem, @NonNull DictPageVO newItem) {
            return oldItem.getId().equals(newItem.getId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull DictPageVO oldItem, @NonNull DictPageVO newItem) {
            return safeEquals(oldItem.getName(), newItem.getName()) &&
                   safeEquals(oldItem.getValue(), newItem.getValue()) &&
                   oldItem.getStatus() == newItem.getStatus();
        }

        private boolean safeEquals(String a, String b) {
            return a == null ? b == null : a.equals(b);
        }
    };

    private OnDictItemActionListener listener;

    public interface OnDictItemActionListener {
        void onEdit(DictPageVO dict);
        void onDelete(DictPageVO dict);
    }

    public void setListener(OnDictItemActionListener listener) {
        this.listener = listener;
    }

    @NonNull
    @Override
    public DictItemViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dict_item, parent, false);
        return new DictItemViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DictItemViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class DictItemViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvName;
        private final TextView tvValue;
        private final TextView tvStatus;
        private final ImageButton btnEdit;
        private final ImageButton btnDelete;

        DictItemViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvValue = itemView.findViewById(R.id.tv_value);
            tvStatus = itemView.findViewById(R.id.tv_status);
            btnEdit = itemView.findViewById(R.id.btn_edit);
            btnDelete = itemView.findViewById(R.id.btn_delete);
        }

        void bind(DictPageVO dict) {
            tvName.setText(dict.getName());
            tvValue.setText(dict.getValue());
            Integer status = dict.getStatus();
            tvStatus.setText(status != null && status == 1 ? "启用" : "禁用");

            btnEdit.setOnClickListener(v -> {
                if (listener != null) listener.onEdit(dict);
            });
            btnDelete.setOnClickListener(v -> {
                if (listener != null) listener.onDelete(dict);
            });
        }
    }
}
