package com.pei.dehaze.ui.system;

import androidx.appcompat.app.AlertDialog;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.Spinner;

import com.pei.dehaze.ui.common.BaseActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityRoleListBinding;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.ui.system.adapter.MenuTreeAdapter;
import com.pei.dehaze.ui.system.adapter.RoleAdapter;
import com.pei.dehaze.ui.system.viewmodel.RoleViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.regex.Pattern;

public class RoleListActivity extends BaseActivity {

    private static final Pattern CODE_PATTERN = Pattern.compile("^[A-Z_]+$");
    private static final String[] DATA_SCOPE_LABELS = {"全部数据", "自定义数据", "本部门数据", "本部门及以下", "仅本人数据"};
    private static final Integer[] DATA_SCOPE_VALUES = {1, 2, 3, 4, 5};

    private RoleViewModel roleViewModel;
    private RoleAdapter roleAdapter;
    private ActivityRoleListBinding binding;

    // 权限分配临时状态
    private RolePageVO pendingPermissionRole;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityRoleListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setupToolbar(binding.toolbar, null);

        roleAdapter = new RoleAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(roleAdapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        roleAdapter.setOnRoleActionListener(new RoleAdapter.OnRoleActionListener() {
            @Override
            public void onEdit(RolePageVO role) {
                if (role.getId() == null) return;
                roleViewModel.loadRoleForm(role.getId());
            }

            @Override
            public void onDelete(RolePageVO role) {
                showDeleteConfirmDialog(role);
            }

            @Override
            public void onAssignPermissions(RolePageVO role) {
                if (role.getId() == null) return;
                pendingPermissionRole = role;
                roleViewModel.clearMenuList();
                roleViewModel.clearRoleMenuIds();
                roleViewModel.loadMenuList();
                roleViewModel.loadRoleMenuIds(role.getId());
            }

            @Override
            public void onToggleStatus(RolePageVO role) {
                if (role.getId() == null) return;
                EnableStatus newStatus = role.getStatus() == EnableStatus.ENABLED
                        ? EnableStatus.DISABLED : EnableStatus.ENABLED;
                roleViewModel.updateRoleStatus(role.getId(), newStatus);
            }
        });

        roleAdapter.setOnSelectionChangedListener(selectedIds ->
                binding.tvPageInfo.setText("已选中 " + selectedIds.size() + " 项"));

        binding.btnSearch.setOnClickListener(v -> {
            String keywords = binding.etKeywords.getText().toString().trim();
            roleViewModel.setKeywords(keywords);
            loadData();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            roleViewModel.resetQuery();
            loadData();
        });

        binding.btnAdd.setOnClickListener(v -> showRoleFormDialog(null));

        binding.btnBatchDelete.setOnClickListener(v -> {
            if (!roleAdapter.isSelectionMode()) {
                roleAdapter.setSelectionMode(true);
                updateSelectionModeUI(true);
                ToastUtils.showShort(this, "长按或勾选要删除的角色");
            } else {
                showBatchDeleteConfirmDialog();
            }
        });

        binding.btnCancelSelect.setOnClickListener(v -> {
            roleAdapter.clearSelection();
            roleAdapter.setSelectionMode(false);
            updateSelectionModeUI(false);
            updatePageInfo();
        });

        binding.btnSelectAll.setOnClickListener(v -> roleAdapter.selectAll());

        binding.btnPrev.setOnClickListener(v -> roleViewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> roleViewModel.nextPage());
    }

    private void updateSelectionModeUI(boolean selectionMode) {
        binding.btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        if (selectionMode) {
            binding.btnBatchDelete.setText("删除选中");
        } else {
            binding.btnBatchDelete.setText("批量删除");
        }
    }

    private void initViewModel() {
        roleViewModel = new ViewModelProvider(this).get(RoleViewModel.class);
    }

    private void setupObservers() {
        roleViewModel.getRoleList().observe(this, roles -> {
            roleAdapter.submitList(roles);
            updatePageInfo();
            binding.tvEmpty.setVisibility(roles == null || roles.isEmpty() ? View.VISIBLE : View.GONE);
        });

        roleViewModel.getTotal().observe(this, total -> updatePageInfo());

        roleViewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        roleViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                roleViewModel.clearError();
            }
        });

        roleViewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                roleViewModel.clearOperationResult();
                if (result.startsWith("删除") || result.startsWith("新增") || result.startsWith("修改")
                        || result.startsWith("状态") || result.startsWith("权限")) {
                    if (roleAdapter.isSelectionMode()) {
                        roleAdapter.setSelectionMode(false);
                        updateSelectionModeUI(false);
                    }
                }
            }
        });

        roleViewModel.getRoleForm().observe(this, form -> {
            if (form != null) {
                showRoleFormDialog(form);
                roleViewModel.clearRoleForm();
            }
        });

        roleViewModel.getMenuList().observe(this, menus -> tryShowPermissionDialog());

        roleViewModel.getRoleMenuIds().observe(this, menuIds -> tryShowPermissionDialog());
    }

    private void tryShowPermissionDialog() {
        if (pendingPermissionRole == null) return;
        List<MenuVO> menus = roleViewModel.getMenuList().getValue();
        List<Integer> menuIds = roleViewModel.getRoleMenuIds().getValue();
        if (menus == null || menuIds == null) return;
        RolePageVO role = pendingPermissionRole;
        pendingPermissionRole = null;
        showPermissionTreeDialog(role, menus, menuIds);
    }

    private void loadData() {
        roleViewModel.loadRoles();
    }

    private void updatePageInfo() {
        if (roleAdapter != null && roleAdapter.isSelectionMode()) {
            int count = roleAdapter.getSelectedIds().size();
            binding.tvPageInfo.setText("已选中 " + count + " 项");
            return;
        }
        long total = roleViewModel.getTotal().getValue() != null ? roleViewModel.getTotal().getValue() : 0L;
        int pageNum = roleViewModel.getPageNum();
        int pageSize = roleViewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(total * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + total + " 条)");
    }

    private void showDeleteConfirmDialog(RolePageVO role) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除角色「" + StringUtils.safe(role.getName()) + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    if (role.getId() != null) {
                        roleViewModel.deleteRoles(Collections.singletonList(role.getId().longValue()));
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showBatchDeleteConfirmDialog() {
        Set<Integer> selectedIds = roleAdapter.getSelectedIds();
        if (selectedIds.isEmpty()) {
            ToastUtils.showShort(this, "请先选择要删除的角色");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("批量删除确认")
                .setMessage("确认删除选中的 " + selectedIds.size() + " 个角色吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        roleViewModel.deleteRoles(selectedIds.stream().map(Integer::longValue).collect(Collectors.toList())))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showRoleFormDialog(RoleForm existingForm) {
        boolean isEdit = existingForm != null && existingForm.getId() != null;
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_role_form, null);

        EditText etName = view.findViewById(R.id.et_name);
        EditText etCode = view.findViewById(R.id.et_code);
        Spinner spinnerDataScope = view.findViewById(R.id.spinner_data_scope);
        EditText etSort = view.findViewById(R.id.et_sort);
        RadioGroup rgStatus = view.findViewById(R.id.rg_status);

        ArrayAdapter<String> dataScopeAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, DATA_SCOPE_LABELS);
        dataScopeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerDataScope.setAdapter(dataScopeAdapter);

        if (isEdit) {
            etName.setText(StringUtils.safe(existingForm.getName()));
            etCode.setText(StringUtils.safe(existingForm.getCode()));
            etCode.setEnabled(false);
            etSort.setText(existingForm.getSort() != null ? String.valueOf(existingForm.getSort()) : "1");
            if (existingForm.getDataScope() != null) {
                for (int i = 0; i < DATA_SCOPE_VALUES.length; i++) {
                    if (DATA_SCOPE_VALUES[i].equals(existingForm.getDataScope())) {
                        spinnerDataScope.setSelection(i);
                        break;
                    }
                }
            } else {
                spinnerDataScope.setSelection(0);
            }
            if (existingForm.getStatus() == EnableStatus.DISABLED) {
                rgStatus.check(R.id.rb_status_disable);
            } else {
                rgStatus.check(R.id.rb_status_enable);
            }
        } else {
            etSort.setText("1");
            spinnerDataScope.setSelection(0);
            rgStatus.check(R.id.rb_status_enable);
        }

        new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改角色" : "新增角色")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = etName.getText().toString().trim();
                    String code = etCode.getText().toString().trim();
                    String sortStr = etSort.getText().toString().trim();
                    int dataScope = DATA_SCOPE_VALUES[spinnerDataScope.getSelectedItemPosition()];
                    EnableStatus status = rgStatus.getCheckedRadioButtonId() == R.id.rb_status_enable
                            ? EnableStatus.ENABLED : EnableStatus.DISABLED;

                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(this, "请输入角色名称");
                        return;
                    }
                    if (TextUtils.isEmpty(code)) {
                        ToastUtils.showShort(this, "请输入角色编码");
                        return;
                    }
                    if (!CODE_PATTERN.matcher(code).matches()) {
                        ToastUtils.showShort(this, "角色编码只能包含大写字母和下划线");
                        return;
                    }
                    int sort;
                    try {
                        sort = Integer.parseInt(sortStr);
                    } catch (NumberFormatException e) {
                        ToastUtils.showShort(this, "排序必须为数字");
                        return;
                    }

                    RoleForm form = new RoleForm();
                    form.setName(name);
                    form.setCode(code);
                    form.setDataScope(dataScope);
                    form.setSort(sort);
                    form.setStatus(status);

                    if (isEdit) {
                        form.setId(existingForm.getId());
                        roleViewModel.updateRole(existingForm.getId(), form);
                    } else {
                        roleViewModel.addRole(form);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showPermissionTreeDialog(RolePageVO role, List<MenuVO> menus, List<Integer> checkedIds) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_permission_tree, null);
        RecyclerView rvMenuTree = view.findViewById(R.id.rv_menu_tree);
        rvMenuTree.setLayoutManager(new LinearLayoutManager(this));

        MenuTreeAdapter menuTreeAdapter = new MenuTreeAdapter();
        rvMenuTree.setAdapter(menuTreeAdapter);
        menuTreeAdapter.setData(menus);
        menuTreeAdapter.setCheckedIds(checkedIds);

        new AlertDialog.Builder(this)
                .setTitle("权限分配 - " + StringUtils.safe(role.getName()))
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    if (role.getId() == null) return;
                    List<Integer> selectedMenuIds = menuTreeAdapter.getCheckedIds();
                    roleViewModel.assignMenus(role.getId(), selectedMenuIds);
                })
                .setNegativeButton("取消", null)
                .show();
    }

}
