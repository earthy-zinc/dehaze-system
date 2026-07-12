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
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;

public class DictTypeAdapter extends ListAdapter<DictTypePageVO, DictTypeAdapter.DictTypeViewHolder> {

    public DictTypeAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<DictTypePageVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<DictTypePageVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull DictTypePageVO oldItem, @NonNull DictTypePageVO newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull DictTypePageVO oldItem, @NonNull DictTypePageVO newItem) {
            return safeEquals(oldItem.getName(), newItem.getName()) &&
                   safeEquals(oldItem.getCode(), newItem.getCode()) &&
                   oldItem.getStatus() == newItem.getStatus() &&
                   safeEquals(oldItem.getRemark(), newItem.getRemark());
        }

        private boolean safeEquals(String a, String b) {
            return a == null ? b == null : a.equals(b);
        }
    };

    private OnDictTypeActionListener listener;

    public interface OnDictTypeActionListener {
        void onEdit(DictTypePageVO dictType);
        void onDelete(DictTypePageVO dictType);
        void onManageItems(DictTypePageVO dictType);
    }

    public void setListener(OnDictTypeActionListener listener) {
        this.listener = listener;
    }

    @NonNull
    @Override
    public DictTypeViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dict_type, parent, false);
        return new DictTypeViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DictTypeViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class DictTypeViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvName;
        private final TextView tvCode;
        private final TextView tvStatus;
        private final TextView tvRemark;
        private final ImageButton btnItems;
        private final ImageButton btnEdit;
        private final ImageButton btnDelete;

        DictTypeViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvCode = itemView.findViewById(R.id.tv_code);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvRemark = itemView.findViewById(R.id.tv_remark);
            btnItems = itemView.findViewById(R.id.btn_items);
            btnEdit = itemView.findViewById(R.id.btn_edit);
            btnDelete = itemView.findViewById(R.id.btn_delete);
        }

        void bind(DictTypePageVO dictType) {
            tvName.setText(dictType.getName());
            tvCode.setText(dictType.getCode());
            Integer status = dictType.getStatus();
            tvStatus.setText(status != null && status == 1 ? "启用" : "禁用");
            tvRemark.setText(dictType.getRemark() != null ? dictType.getRemark() : "");

            btnItems.setOnClickListener(v -> {
                if (listener != null) listener.onManageItems(dictType);
            });
            btnEdit.setOnClickListener(v -> {
                if (listener != null) listener.onEdit(dictType);
            });
            btnDelete.setOnClickListener(v -> {
                if (listener != null) listener.onDelete(dictType);
            });
        }
    }
}
