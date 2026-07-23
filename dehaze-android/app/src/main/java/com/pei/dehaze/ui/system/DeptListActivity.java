package com.pei.dehaze.ui.system;

import android.app.AlertDialog;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.button.MaterialButton;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptVO;
import com.pei.dehaze.ui.system.adapter.DeptAdapter;
import com.pei.dehaze.ui.system.viewmodel.DeptViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DeptListActivity extends AppCompatActivity {

    private static final String[] STATUS_LABELS = {"全部", "启用", "禁用"};
    private static final Integer[] STATUS_VALUES = {null, 1, 0};

    private DeptViewModel deptViewModel;
    private DeptAdapter deptAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    private Toolbar toolbar;
    private TextView tvEmpty;
    private EditText etKeywords;
    private Spinner spinnerStatus;
    private MaterialButton btnSearch;
    private MaterialButton btnReset;
    private MaterialButton btnAdd;

    private final List<Option> deptOptions = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dept_list);

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
        btnSearch = findViewById(R.id.btn_search);
        btnReset = findViewById(R.id.btn_reset);
        btnAdd = findViewById(R.id.btn_add);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        deptAdapter = new DeptAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(deptAdapter);

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, STATUS_LABELS);
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerStatus.setAdapter(statusAdapter);

        swipeRefreshLayout.setOnRefreshListener(this::loadData);

        deptAdapter.setOnDeptActionListener(new DeptAdapter.OnDeptActionListener() {
            @Override
            public void onEdit(DeptVO dept) {
                if (dept.getId() == null) return;
                deptViewModel.loadDeptForm(dept.getId());
            }

            @Override
            public void onDelete(DeptVO dept) {
                showDeleteConfirmDialog(dept);
            }

            @Override
            public void onAddChild(DeptVO dept) {
                showDeptFormDialog(null, dept);
            }
        });

        btnSearch.setOnClickListener(v -> {
            String keywords = etKeywords.getText().toString().trim();
            Integer status = STATUS_VALUES[spinnerStatus.getSelectedItemPosition()];
            deptViewModel.setQueryParams(keywords, status);
            loadData();
        });

        btnReset.setOnClickListener(v -> {
            etKeywords.setText("");
            spinnerStatus.setSelection(0);
            deptViewModel.resetQuery();
            loadData();
        });

        btnAdd.setOnClickListener(v -> showDeptFormDialog(null, null));
    }

    private void initViewModel() {
        deptViewModel = new ViewModelProvider(this).get(DeptViewModel.class);
    }

    private void setupObservers() {
        deptViewModel.getDeptList().observe(this, depts -> {
            deptAdapter.setData(depts);
            tvEmpty.setVisibility(depts == null || depts.isEmpty() ? View.VISIBLE : View.GONE);
        });

        deptViewModel.getLoading().observe(this, isLoading ->
                swipeRefreshLayout.setRefreshing(isLoading != null && isLoading));

        deptViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                deptViewModel.clearError();
            }
        });

        deptViewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                deptViewModel.clearOperationResult();
            }
        });

        deptViewModel.getDeptForm().observe(this, form -> {
            if (form != null) {
                showDeptFormDialog(form, null);
                deptViewModel.clearDeptForm();
            }
        });

        deptViewModel.getDeptOptions().observe(this, options -> {
            deptOptions.clear();
            if (options != null) {
                deptOptions.addAll(options);
            }
        });
    }

    private void loadData() {
        deptViewModel.loadDepts();
        deptViewModel.loadDeptOptions();
    }

    private void showDeleteConfirmDialog(DeptVO dept) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除部门「" + StringUtils.safe(dept.getName()) + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    if (dept.getId() != null) {
                        deptViewModel.deleteDepts(Collections.singletonList(dept.getId().longValue()));
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showDeptFormDialog(DeptForm existingForm, DeptVO parentDept) {
        boolean isEdit = existingForm != null && existingForm.getId() != null;
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_dept_form, null);

        EditText etName = view.findViewById(R.id.et_name);
        TextView tvParent = view.findViewById(R.id.tv_parent);
        EditText etSort = view.findViewById(R.id.et_sort);
        RadioGroup rgStatus = view.findViewById(R.id.rg_status);

        // 临时状态：选中的父部门 id 与名称
        int[] selectedParentId = {0};
        String[] selectedParentName = {null};

        if (isEdit) {
            etName.setText(StringUtils.safe(existingForm.getName()));
            etSort.setText(existingForm.getSort() != null ? String.valueOf(existingForm.getSort()) : "1");
            selectedParentId[0] = existingForm.getParentId();
            for (Option opt : deptOptions) {
                if (String.valueOf(existingForm.getParentId()).equals(opt.getValue())) {
                    selectedParentName[0] = opt.getLabel();
                    break;
                }
            }
            tvParent.setText(selectedParentName[0] == null
                    ? (existingForm.getParentId() == 0 ? "无（顶级部门）" : String.valueOf(existingForm.getParentId()))
                    : selectedParentName[0]);
            if (existingForm.getStatus() != null && existingForm.getStatus() == 0) {
                rgStatus.check(R.id.rb_status_disable);
            } else {
                rgStatus.check(R.id.rb_status_enable);
            }
        } else if (parentDept != null) {
            // 新增子部门
            etSort.setText("1");
            rgStatus.check(R.id.rb_status_enable);
            selectedParentId[0] = parentDept.getId() != null ? parentDept.getId() : 0;
            selectedParentName[0] = StringUtils.safe(parentDept.getName());
            tvParent.setText(selectedParentName[0]);
        } else {
            // 新增顶级部门
            etSort.setText("1");
            rgStatus.check(R.id.rb_status_enable);
            tvParent.setText("无（顶级部门）");
        }

        tvParent.setOnClickListener(v -> showParentPickerDialog(selected -> {
            selectedParentId[0] = StringUtils.safeParseInt(selected.getValue(), 0);
            selectedParentName[0] = selected.getLabel();
            tvParent.setText(selectedParentName[0]);
        }));

        new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改部门" : "新增部门")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = etName.getText().toString().trim();
                    String sortStr = etSort.getText().toString().trim();
                    int status = rgStatus.getCheckedRadioButtonId() == R.id.rb_status_enable ? 1 : 0;

                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(this, "请输入部门名称");
                        return;
                    }
                    int sort;
                    try {
                        sort = Integer.parseInt(sortStr);
                    } catch (NumberFormatException e) {
                        ToastUtils.showShort(this, "排序必须为数字");
                        return;
                    }

                    DeptForm form = new DeptForm();
                    form.setName(name);
                    form.setParentId(selectedParentId[0]);
                    form.setSort(sort);
                    form.setStatus(status);

                    if (isEdit) {
                        form.setId(existingForm.getId());
                        deptViewModel.updateDept(existingForm.getId(), form);
                    } else {
                        deptViewModel.addDept(form);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showParentPickerDialog(OnOptionSelectedListener listener) {
        // 构造选项列表，首项为"无（顶级部门）"
        List<Option> options = new ArrayList<>();
        Option none = new Option();
        none.setLabel("无（顶级部门）");
        none.setValue("0");
        options.add(none);
        options.addAll(deptOptions);

        String[] items = new String[options.size()];
        for (int i = 0; i < options.size(); i++) {
            items[i] = options.get(i).getLabel();
        }
        new AlertDialog.Builder(this)
                .setTitle("选择上级部门")
                .setItems(items, (dialog, which) -> {
                    if (which >= 0 && which < options.size()) {
                        listener.onSelected(options.get(which));
                    }
                })
                .show();
    }

    private interface OnOptionSelectedListener {
        void onSelected(Option option);
    }

}
