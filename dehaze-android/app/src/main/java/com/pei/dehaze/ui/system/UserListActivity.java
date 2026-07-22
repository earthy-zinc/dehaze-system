package com.pei.dehaze.ui.system;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.button.MaterialButton;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.user.UserForm;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.ui.system.adapter.UserAdapter;
import com.pei.dehaze.ui.system.viewmodel.UserViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

public class UserListActivity extends AppCompatActivity {

    private static final int REQUEST_STORAGE_PERMISSION = 1001;
    private static final int REQUEST_IMPORT_FILE = 1002;
    private static final Pattern MOBILE_PATTERN = Pattern.compile("^1[3-9]\\d{9}$");
    private static final Pattern EMAIL_PATTERN = Pattern.compile("^\\w+([-+.]\\w+)*@\\w+([-.]\\w+)*\\.\\w+([-.]\\w+)*$");

    private UserViewModel userViewModel;
    private UserAdapter userAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    private Toolbar toolbar;
    private TextView tvEmpty;
    private EditText etKeywords;
    private Spinner spinnerStatus;
    private Spinner spinnerDept;
    private MaterialButton btnSearch;
    private MaterialButton btnReset;
    private MaterialButton btnAdd;
    private MaterialButton btnBatchDelete;
    private MaterialButton btnCancelSelect;
    private MaterialButton btnSelectAll;
    private MaterialButton btnExport;
    private MaterialButton btnTemplate;
    private MaterialButton btnPrev;
    private MaterialButton btnNext;
    private TextView tvPageInfo;

    private final List<Option> deptOptions = new ArrayList<>();
    private final List<Option> roleOptions = new ArrayList<>();
    private static final String[] STATUS_LABELS = {"全部", "启用", "禁用"};
    private static final Integer[] STATUS_VALUES = {null, 1, 0};
    private static final String[] GENDER_LABELS = {"未知", "男", "女"};
    private static final Integer[] GENDER_VALUES = {0, 1, 2};

    // 表单临时状态
    private Integer selectedDeptId;
    private String selectedDeptName;
    private final Set<Integer> selectedRoleIds = new HashSet<>();
    private final List<String> selectedRoleNames = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_list);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        tvEmpty = findViewById(R.id.tv_empty);
        etKeywords = findViewById(R.id.et_keywords);
        spinnerStatus = findViewById(R.id.spinner_status);
        spinnerDept = findViewById(R.id.spinner_dept);
        btnSearch = findViewById(R.id.btn_search);
        btnReset = findViewById(R.id.btn_reset);
        btnAdd = findViewById(R.id.btn_add);
        btnBatchDelete = findViewById(R.id.btn_batch_delete);
        btnCancelSelect = findViewById(R.id.btn_cancel_select);
        btnSelectAll = findViewById(R.id.btn_select_all);
        btnExport = findViewById(R.id.btn_export);
        btnTemplate = findViewById(R.id.btn_template);
        btnPrev = findViewById(R.id.btn_prev);
        btnNext = findViewById(R.id.btn_next);
        tvPageInfo = findViewById(R.id.tv_page_info);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        userAdapter = new UserAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(userAdapter);

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, STATUS_LABELS);
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerStatus.setAdapter(statusAdapter);

        ArrayAdapter<String> deptAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, new ArrayList<>(java.util.Collections.singletonList("全部部门")));
        deptAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerDept.setAdapter(deptAdapter);

        swipeRefreshLayout.setOnRefreshListener(this::loadData);

        userAdapter.setOnUserActionListener(new UserAdapter.OnUserActionListener() {
            @Override
            public void onEdit(UserPageVO user) {
                userViewModel.loadUserForm(user.getId());
            }

            @Override
            public void onDelete(UserPageVO user) {
                showDeleteConfirmDialog(user);
            }

            @Override
            public void onResetPassword(UserPageVO user) {
                showResetPasswordDialog(user);
            }

            @Override
            public void onToggleStatus(UserPageVO user) {
                if (user.getId() == null) return;
                int newStatus = (user.getStatus() != null && user.getStatus() == 1) ? 0 : 1;
                userViewModel.updateUserStatus(user.getId(), newStatus);
            }
        });

        userAdapter.setOnSelectionChangedListener(selectedIds -> {
            tvPageInfo.setText("已选中 " + selectedIds.size() + " 项");
        });

        btnSearch.setOnClickListener(v -> {
            String keywords = etKeywords.getText().toString().trim();
            Integer status = STATUS_VALUES[spinnerStatus.getSelectedItemPosition()];
            Integer deptId = null;
            int deptPos = spinnerDept.getSelectedItemPosition();
            if (deptPos > 0 && deptPos - 1 < deptOptions.size()) {
                try {
                    deptId = Integer.parseInt(deptOptions.get(deptPos - 1).getValue());
                } catch (NumberFormatException ignored) {
                }
            }
            userViewModel.setQueryParams(keywords, status, deptId, null, null);
            loadData();
        });

        btnReset.setOnClickListener(v -> {
            etKeywords.setText("");
            spinnerStatus.setSelection(0);
            if (spinnerDept.getAdapter() != null && spinnerDept.getAdapter().getCount() > 0) {
                spinnerDept.setSelection(0);
            }
            userViewModel.resetQuery();
            loadData();
        });

        btnAdd.setOnClickListener(v -> showUserFormDialog(null));
        btnBatchDelete.setOnClickListener(v -> {
            if (!userAdapter.isSelectionMode()) {
                userAdapter.setSelectionMode(true);
                updateSelectionModeUI(true);
                ToastUtils.showShort(this, "长按或勾选要删除的用户");
            } else {
                showBatchDeleteConfirmDialog();
            }
        });
        btnCancelSelect.setOnClickListener(v -> {
            userAdapter.clearSelection();
            userAdapter.setSelectionMode(false);
            updateSelectionModeUI(false);
            updatePageInfo();
        });
        btnSelectAll.setOnClickListener(v -> userAdapter.selectAll());

        btnExport.setOnClickListener(v -> {
            if (!ensureStoragePermission()) return;
            File dir = new File(getExternalFilesDir(null), "exports");
            if (!dir.exists() && !dir.mkdirs()) {
                ToastUtils.showShort(this, "无法创建导出目录");
                return;
            }
            String fileName = "users_" + System.currentTimeMillis() + ".xlsx";
            File file = new File(dir, fileName);
            userViewModel.exportUsers(file.getAbsolutePath());
        });

        btnTemplate.setOnClickListener(v -> {
            if (!ensureStoragePermission()) return;
            File dir = new File(getExternalFilesDir(null), "templates");
            if (!dir.exists() && !dir.mkdirs()) {
                ToastUtils.showShort(this, "无法创建模板目录");
                return;
            }
            File file = new File(dir, "user_import_template.xlsx");
            userViewModel.downloadTemplate(file.getAbsolutePath());
        });

        btnPrev.setOnClickListener(v -> userViewModel.prevPage());
        btnNext.setOnClickListener(v -> userViewModel.nextPage());
    }

    private void updateSelectionModeUI(boolean selectionMode) {
        btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnExport.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnTemplate.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        if (selectionMode) {
            btnBatchDelete.setText("删除选中");
        } else {
            btnBatchDelete.setText("批量删除");
        }
    }

    private void initViewModel() {
        userViewModel = new ViewModelProvider(this).get(UserViewModel.class);
    }

    private void setupObservers() {
        userViewModel.getUserList().observe(this, users -> {
            userAdapter.submitList(users);
            updatePageInfo();
            tvEmpty.setVisibility(users == null || users.isEmpty() ? View.VISIBLE : View.GONE);
        });

        userViewModel.getTotal().observe(this, total -> updatePageInfo());

        userViewModel.getLoading().observe(this, isLoading ->
                swipeRefreshLayout.setRefreshing(isLoading != null && isLoading));

        userViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                userViewModel.clearError();
            }
        });

        userViewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                userViewModel.clearOperationResult();
                if (result.startsWith("删除") || result.startsWith("新增") || result.startsWith("修改") || result.startsWith("状态")) {
                    if (userAdapter.isSelectionMode()) {
                        userAdapter.setSelectionMode(false);
                        updateSelectionModeUI(false);
                    }
                }
            }
        });

        userViewModel.getUserForm().observe(this, form -> {
            if (form != null) {
                showUserFormDialog(form);
                userViewModel.clearUserForm();
            }
        });

        userViewModel.getDeptOptions().observe(this, options -> {
            deptOptions.clear();
            if (options != null) {
                deptOptions.addAll(options);
            }
            List<String> deptLabels = new ArrayList<>();
            deptLabels.add("全部部门");
            for (Option option : deptOptions) {
                deptLabels.add(option.getLabel());
            }
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_spinner_item, deptLabels);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinnerDept.setAdapter(adapter);
        });

        userViewModel.getRoleOptions().observe(this, options -> {
            roleOptions.clear();
            if (options != null) {
                roleOptions.addAll(options);
            }
        });
    }

    private void loadData() {
        userViewModel.loadUsers();
        userViewModel.loadDeptOptions();
        userViewModel.loadRoleOptions();
    }

    private void updatePageInfo() {
        if (userAdapter != null && userAdapter.isSelectionMode()) {
            int count = userAdapter.getSelectedIds().size();
            tvPageInfo.setText("已选中 " + count + " 项");
            return;
        }
        long total = userViewModel.getTotal().getValue() != null ? userViewModel.getTotal().getValue() : 0L;
        int pageNum = userViewModel.getPageNum();
        int pageSize = userViewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(total * 1.0 / pageSize));
        tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + total + " 条)");
    }

    private void showDeleteConfirmDialog(UserPageVO user) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除用户「" + (user.getUsername() == null ? "" : user.getUsername()) + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    if (user.getId() != null) {
                        userViewModel.deleteUsers(String.valueOf(user.getId()));
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showBatchDeleteConfirmDialog() {
        Set<Integer> selectedIds = userAdapter.getSelectedIds();
        if (selectedIds.isEmpty()) {
            ToastUtils.showShort(this, "请先选择要删除的用户");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("批量删除确认")
                .setMessage("确认删除选中的 " + selectedIds.size() + " 个用户吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    StringBuilder sb = new StringBuilder();
                    for (Integer id : selectedIds) {
                        if (sb.length() > 0) sb.append(",");
                        sb.append(id);
                    }
                    userViewModel.deleteUsers(sb.toString());
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showResetPasswordDialog(UserPageVO user) {
        if (user.getId() == null) return;
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_reset_password, null);
        EditText etPassword = view.findViewById(R.id.et_password);
        EditText etConfirm = view.findViewById(R.id.et_confirm_password);

        new AlertDialog.Builder(this)
                .setTitle("重置密码 - " + (user.getUsername() == null ? "" : user.getUsername()))
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String pwd = etPassword.getText().toString().trim();
                    String confirm = etConfirm.getText().toString().trim();
                    if (TextUtils.isEmpty(pwd)) {
                        ToastUtils.showShort(this, "请输入新密码");
                        return;
                    }
                    if (pwd.length() < 6) {
                        ToastUtils.showShort(this, "密码长度不能少于6位");
                        return;
                    }
                    if (!pwd.equals(confirm)) {
                        ToastUtils.showShort(this, "两次输入的密码不一致");
                        return;
                    }
                    userViewModel.updateUserPassword(user.getId(), pwd);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showUserFormDialog(UserForm existingForm) {
        boolean isEdit = existingForm != null && existingForm.getId() != null;
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_user_form, null);

        EditText etUsername = view.findViewById(R.id.et_username);
        EditText etNickname = view.findViewById(R.id.et_nickname);
        TextView tvDept = view.findViewById(R.id.tv_dept);
        TextView tvRoles = view.findViewById(R.id.tv_roles);
        Spinner spinnerGender = view.findViewById(R.id.spinner_gender);
        EditText etMobile = view.findViewById(R.id.et_mobile);
        EditText etEmail = view.findViewById(R.id.et_email);
        RadioGroup rgStatus = view.findViewById(R.id.rg_status);

        ArrayAdapter<String> genderAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, GENDER_LABELS);
        genderAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerGender.setAdapter(genderAdapter);

        // 重置临时状态
        selectedRoleIds.clear();
        selectedRoleNames.clear();
        selectedDeptId = null;
        selectedDeptName = null;

        if (isEdit) {
            etUsername.setText(existingForm.getUsername() == null ? "" : existingForm.getUsername());
            etUsername.setEnabled(false);
            etNickname.setText(existingForm.getNickname() == null ? "" : existingForm.getNickname());
            etMobile.setText(existingForm.getMobile() == null ? "" : existingForm.getMobile());
            etEmail.setText(existingForm.getEmail() == null ? "" : existingForm.getEmail());
            selectedDeptId = existingForm.getDeptId();
            // 根据 deptId 查找名称
            if (selectedDeptId != null) {
                for (Option opt : deptOptions) {
                    if (String.valueOf(selectedDeptId).equals(opt.getValue())) {
                        selectedDeptName = opt.getLabel();
                        break;
                    }
                }
            }
            tvDept.setText(selectedDeptName == null ? "请选择部门" : selectedDeptName);
            if (existingForm.getRoleIds() != null) {
                selectedRoleIds.addAll(existingForm.getRoleIds());
                for (Integer roleId : selectedRoleIds) {
                    for (Option opt : roleOptions) {
                        if (String.valueOf(roleId).equals(opt.getValue())) {
                            selectedRoleNames.add(opt.getLabel());
                            break;
                        }
                    }
                }
            }
            tvRoles.setText(TextUtils.join(", ", selectedRoleNames));
            if (existingForm.getGender() != null) {
                for (int i = 0; i < GENDER_VALUES.length; i++) {
                    if (GENDER_VALUES[i].equals(existingForm.getGender())) {
                        spinnerGender.setSelection(i);
                        break;
                    }
                }
            } else {
                spinnerGender.setSelection(0);
            }
            if (existingForm.getStatus() != null && existingForm.getStatus() == 0) {
                rgStatus.check(R.id.rb_status_disable);
            } else {
                rgStatus.check(R.id.rb_status_enable);
            }
        } else {
            rgStatus.check(R.id.rb_status_enable);
            spinnerGender.setSelection(0);
            tvDept.setText("请选择部门");
            tvRoles.setText("请选择角色");
        }

        // 部门选择
        tvDept.setOnClickListener(v -> showDeptPickerDialog(deptOptions, selected -> {
            selectedDeptId = selected.getValue() == null ? null : safeParseInt(selected.getValue());
            selectedDeptName = selected.getLabel();
            tvDept.setText(selectedDeptName);
        }));

        // 角色多选
        tvRoles.setOnClickListener(v -> showRoleMultiSelectDialog());

        new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改用户" : "新增用户")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String username = etUsername.getText().toString().trim();
                    String nickname = etNickname.getText().toString().trim();
                    String mobile = etMobile.getText().toString().trim();
                    String email = etEmail.getText().toString().trim();
                    int gender = GENDER_VALUES[spinnerGender.getSelectedItemPosition()];
                    int status = rgStatus.getCheckedRadioButtonId() == R.id.rb_status_enable ? 1 : 0;

                    if (TextUtils.isEmpty(username)) {
                        ToastUtils.showShort(this, "请输入用户名");
                        return;
                    }
                    if (TextUtils.isEmpty(nickname)) {
                        ToastUtils.showShort(this, "请输入昵称");
                        return;
                    }
                    if (selectedDeptId == null) {
                        ToastUtils.showShort(this, "请选择所属部门");
                        return;
                    }
                    if (selectedRoleIds.isEmpty()) {
                        ToastUtils.showShort(this, "请选择角色");
                        return;
                    }
                    if (!TextUtils.isEmpty(mobile) && !MOBILE_PATTERN.matcher(mobile).matches()) {
                        ToastUtils.showShort(this, "手机号格式不正确");
                        return;
                    }
                    if (!TextUtils.isEmpty(email) && !EMAIL_PATTERN.matcher(email).matches()) {
                        ToastUtils.showShort(this, "邮箱格式不正确");
                        return;
                    }

                    UserForm form = new UserForm();
                    form.setUsername(username);
                    form.setNickname(nickname);
                    form.setDeptId(selectedDeptId);
                    form.setGender(gender);
                    form.setRoleIds(new ArrayList<>(selectedRoleIds));
                    form.setMobile(TextUtils.isEmpty(mobile) ? null : mobile);
                    form.setEmail(TextUtils.isEmpty(email) ? null : email);
                    form.setStatus(status);

                    if (isEdit) {
                        form.setId(existingForm.getId());
                        userViewModel.updateUser(existingForm.getId(), form);
                    } else {
                        userViewModel.addUser(form);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showDeptPickerDialog(List<Option> options, OnOptionSelectedListener listener) {
        String[] items = new String[options.size()];
        for (int i = 0; i < options.size(); i++) {
            items[i] = options.get(i).getLabel();
        }
        new AlertDialog.Builder(this)
                .setTitle("选择部门")
                .setItems(items, (dialog, which) -> {
                    if (which >= 0 && which < options.size()) {
                        listener.onSelected(options.get(which));
                    }
                })
                .show();
    }

    private void showRoleMultiSelectDialog() {
        String[] items = new String[roleOptions.size()];
        boolean[] checked = new boolean[roleOptions.size()];
        for (int i = 0; i < roleOptions.size(); i++) {
            Option opt = roleOptions.get(i);
            items[i] = opt.getLabel();
            Integer roleId = safeParseInt(opt.getValue());
            if (roleId != null && selectedRoleIds.contains(roleId)) {
                checked[i] = true;
            }
        }
        new AlertDialog.Builder(this)
                .setTitle("选择角色")
                .setMultiChoiceItems(items, checked, (dialog, which, isChecked) -> {
                    Option opt = roleOptions.get(which);
                    Integer roleId = safeParseInt(opt.getValue());
                    if (roleId == null) return;
                    if (isChecked) {
                        selectedRoleIds.add(roleId);
                    } else {
                        selectedRoleIds.remove(roleId);
                    }
                })
                .setPositiveButton("确定", (dialog, which) -> {
                    selectedRoleNames.clear();
                    for (Integer roleId : selectedRoleIds) {
                        for (Option opt : roleOptions) {
                            if (String.valueOf(roleId).equals(opt.getValue())) {
                                selectedRoleNames.add(opt.getLabel());
                                break;
                            }
                        }
                    }
                    // 更新 tvRoles 文本需要重新展示对话框外的 TextView，但这里 dialog 已 dismiss
                    // 改为通过回调更新
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private interface OnOptionSelectedListener {
        void onSelected(Option option);
    }

    private Integer safeParseInt(String value) {
        if (value == null) return null;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private boolean ensureStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE},
                        REQUEST_STORAGE_PERMISSION);
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            boolean granted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            if (!granted) {
                ToastUtils.showShort(this, "存储权限被拒绝，无法操作文件");
            } else {
                ToastUtils.showShort(this, "权限已授予，请重新点击操作");
            }
        }
    }
}
