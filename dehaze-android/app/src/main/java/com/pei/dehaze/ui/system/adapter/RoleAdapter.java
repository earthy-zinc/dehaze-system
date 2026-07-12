package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.role.RolePageVO;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class RoleAdapter extends ListAdapter<RolePageVO, RoleAdapter.RoleViewHolder> {

    public interface OnRoleActionListener {
        void onEdit(RolePageVO role);
        void onDelete(RolePageVO role);
        void onAssignPermissions(RolePageVO role);
        void onToggleStatus(RolePageVO role);
    }

    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Integer> selectedIds);
    }

    private OnRoleActionListener actionListener;
    private OnSelectionChangedListener selectionListener;
    private boolean selectionMode = false;
    private final Set<Integer> selectedIds = new HashSet<>();
    private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());

    public RoleAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<RolePageVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<RolePageVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull RolePageVO oldItem, @NonNull RolePageVO newItem) {
            return oldItem.getId() != null && oldItem.getId().equals(newItem.getId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull RolePageVO oldItem, @NonNull RolePageVO newItem) {
            return RoleAdapter.equals(oldItem.getName(), newItem.getName()) &&
                   RoleAdapter.equals(oldItem.getCode(), newItem.getCode()) &&
                   RoleAdapter.equals(oldItem.getStatus(), newItem.getStatus()) &&
                   RoleAdapter.equals(oldItem.getSort(), newItem.getSort());
        }
    };

    private static boolean equals(Object a, Object b) {
        return a == null ? b == null : a.equals(b);
    }

    public void setOnRoleActionListener(OnRoleActionListener listener) {
        this.actionListener = listener;
    }

    public void setOnSelectionChangedListener(OnSelectionChangedListener listener) {
        this.selectionListener = listener;
    }

    public void setSelectionMode(boolean selectionMode) {
        this.selectionMode = selectionMode;
        if (!selectionMode) {
            selectedIds.clear();
        }
        notifyItemRangeChanged(0, getItemCount());
    }

    public boolean isSelectionMode() {
        return selectionMode;
    }

    public void selectAll() {
        for (RolePageVO role : getCurrentList()) {
            if (role.getId() != null) {
                selectedIds.add(role.getId());
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

    public Set<Integer> getSelectedIds() {
        return new HashSet<>(selectedIds);
    }

    public List<Integer> getSelectedIdList() {
        return new ArrayList<>(selectedIds);
    }

    public String getSelectedIdsString() {
        if (selectedIds.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (Integer id : selectedIds) {
            if (sb.length() > 0) {
                sb.append(",");
            }
            sb.append(id);
        }
        return sb.toString();
    }

    private void notifySelectionChanged() {
        if (selectionListener != null) {
            selectionListener.onSelectionChanged(new HashSet<>(selectedIds));
        }
    }

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

    class RoleViewHolder extends RecyclerView.ViewHolder {
        private CheckBox cbSelect;
        private TextView tvName;
        private TextView tvCode;
        private TextView tvSort;
        private TextView tvStatus;
        private TextView tvCreateTime;
        private TextView tvEdit;
        private TextView tvDelete;
        private TextView tvAssignPermissions;
        private TextView tvToggleStatus;

        RoleViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            tvName = itemView.findViewById(R.id.tv_name);
            tvCode = itemView.findViewById(R.id.tv_code);
            tvSort = itemView.findViewById(R.id.tv_sort);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvAssignPermissions = itemView.findViewById(R.id.tv_assign_permissions);
            tvToggleStatus = itemView.findViewById(R.id.tv_toggle_status);
        }

        void bind(RolePageVO role) {
            tvName.setText(safe(role.getName()));
            tvCode.setText(safe(role.getCode()));
            tvSort.setText(role.getSort() != null ? String.valueOf(role.getSort()) : "0");
            Integer status = role.getStatus();
            if (status != null && status == 1) {
                tvStatus.setText("启用");
                tvStatus.setTextColor(0xFF4CAF50);
                tvToggleStatus.setText("禁用");
            } else {
                tvStatus.setText("禁用");
                tvStatus.setTextColor(0xFF9E9E9E);
                tvToggleStatus.setText("启用");
            }
            Date createTime = role.getCreateTime();
            tvCreateTime.setText(createTime != null ? dateFormat.format(createTime) : "-");

            if (selectionMode) {
                cbSelect.setVisibility(View.VISIBLE);
                cbSelect.setChecked(role.getId() != null && selectedIds.contains(role.getId()));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (role.getId() == null) {
                        cbSelect.setChecked(false);
                        return;
                    }
                    if (checked) {
                        selectedIds.add(role.getId());
                    } else {
                        selectedIds.remove(role.getId());
                    }
                    notifySelectionChanged();
                });
                hideActions();
            } else {
                cbSelect.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                showActions();
            }

            tvEdit.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onEdit(role);
            });
            tvDelete.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onDelete(role);
            });
            tvAssignPermissions.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onAssignPermissions(role);
            });
            tvToggleStatus.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onToggleStatus(role);
            });

            itemView.setOnLongClickListener(v -> {
                if (!selectionMode && role.getId() != null) {
                    setSelectionMode(true);
                    selectedIds.add(role.getId());
                    notifyItemRangeChanged(0, getItemCount());
                    notifySelectionChanged();
                    return true;
                }
                return false;
            });
        }

        private void showActions() {
            tvEdit.setVisibility(View.VISIBLE);
            tvDelete.setVisibility(View.VISIBLE);
            tvAssignPermissions.setVisibility(View.VISIBLE);
            tvToggleStatus.setVisibility(View.VISIBLE);
        }

        private void hideActions() {
            tvEdit.setVisibility(View.GONE);
            tvDelete.setVisibility(View.GONE);
            tvAssignPermissions.setVisibility(View.GONE);
            tvToggleStatus.setVisibility(View.GONE);
        }

        private String safe(String s) {
            return s == null ? "" : s;
        }
    }
}
