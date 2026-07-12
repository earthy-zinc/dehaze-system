package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.ui.system.adapter.MenuAdapter;
import com.pei.dehaze.ui.system.viewmodel.MenuViewModel;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.List;

public class MenuListActivity extends AppCompatActivity {

    private MenuViewModel menuViewModel;
    private MenuAdapter menuAdapter;
    private SwipeRefreshLayout swipeRefreshLayout;
    private TextInputEditText etSearch;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu_list);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        MaterialToolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());

        etSearch = findViewById(R.id.et_search);
        RecyclerView recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);

        menuAdapter = new MenuAdapter();
        menuAdapter.setListener(new MenuAdapter.OnMenuActionListener() {
            @Override
            public void onEdit(MenuVO menu) {
                menuViewModel.loadMenuForm(menu.getId());
            }

            @Override
            public void onDelete(MenuVO menu) {
                confirmDelete(menu);
            }
        });
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(menuAdapter);

        swipeRefreshLayout.setOnRefreshListener(() -> menuViewModel.loadMenus(null));

        MaterialButton btnSearch = findViewById(R.id.btn_search);
        MaterialButton btnReset = findViewById(R.id.btn_reset);
        MaterialButton btnAdd = findViewById(R.id.btn_add);

        btnSearch.setOnClickListener(v -> {
            String keywords = etSearch.getText() != null ? etSearch.getText().toString().trim() : "";
            menuViewModel.loadMenus(keywords.isEmpty() ? null : keywords);
        });

        btnReset.setOnClickListener(v -> {
            etSearch.setText("");
            menuViewModel.loadMenus(null);
        });

        btnAdd.setOnClickListener(v -> showFormDialog(null));
    }

    private void initViewModel() {
        menuViewModel = new MenuViewModel();
    }

    private void setupObservers() {
        menuViewModel.getMenuList().observe(this, menus -> menuAdapter.setMenuTree(menus));

        menuViewModel.getLoading().observe(this, isLoading ->
                swipeRefreshLayout.setRefreshing(isLoading != null && isLoading));

        menuViewModel.getError().observe(this, errorMsg -> {
            if (!TextUtils.isEmpty(errorMsg)) {
                Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show();
            }
        });

        menuViewModel.getActionResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                Toast.makeText(this, result, Toast.LENGTH_SHORT).show();
            }
        });

        menuViewModel.getMenuForm().observe(this, form -> showFormDialog(form));
    }

    private void loadData() {
        menuViewModel.loadMenus(null);
    }

    private void confirmDelete(MenuVO menu) {
        new AlertDialog.Builder(this)
                .setTitle("确认删除")
                .setMessage("确认删除菜单「" + menu.getName() + "」吗？删除后不可恢复。")
                .setPositiveButton("确认", (dialog, which) -> menuViewModel.deleteMenu(menu.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showFormDialog(MenuForm existingForm) {
        boolean isEdit = existingForm != null;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_menu_form, null);

        RadioGroup rgType = formView.findViewById(R.id.rg_type);
        TextInputEditText etParentId = formView.findViewById(R.id.et_parent_id);
        TextInputEditText etName = formView.findViewById(R.id.et_name);
        TextInputEditText etPath = formView.findViewById(R.id.et_path);
        TextInputEditText etComponent = formView.findViewById(R.id.et_component);
        TextInputEditText etPerm = formView.findViewById(R.id.et_perm);
        TextInputEditText etIcon = formView.findViewById(R.id.et_icon);
        TextInputEditText etRedirect = formView.findViewById(R.id.et_redirect);
        TextInputEditText etSort = formView.findViewById(R.id.et_sort);
        RadioGroup rgVisible = formView.findViewById(R.id.rg_visible);

        if (isEdit) {
            selectTypeRadio(rgType, existingForm.getType());
            etParentId.setText(existingForm.getParentId() != null
                    ? String.valueOf(existingForm.getParentId()) : "0");
            etName.setText(existingForm.getName());
            etPath.setText(existingForm.getPath());
            etComponent.setText(existingForm.getComponent());
            etPerm.setText(existingForm.getPerm());
            etIcon.setText(existingForm.getIcon());
            etRedirect.setText(existingForm.getRedirect());
            etSort.setText(String.valueOf(existingForm.getSort()));
            selectVisibleRadio(rgVisible, existingForm.getVisible());
        } else {
            etParentId.setText("0");
            etSort.setText("1");
        }

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改菜单" : "新增菜单")
                .setView(formView)
                .setPositiveButton("确定", null)
                .setNegativeButton("取消", null)
                .create();

        dialog.setOnShowListener(d -> dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setOnClickListener(v -> {
                    String name = getText(etName);
                    if (TextUtils.isEmpty(name)) {
                        Toast.makeText(this, "菜单名称不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    String sortStr = getText(etSort);
                    if (TextUtils.isEmpty(sortStr)) {
                        Toast.makeText(this, "排序不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    MenuForm form = buildFormFromView(rgType, etParentId, etName, etPath,
                            etComponent, etPerm, etIcon, etRedirect, etSort, rgVisible,
                            isEdit ? existingForm.getId() : null);

                    if (isEdit) {
                        menuViewModel.updateMenu(Long.parseLong(form.getId()), form);
                    } else {
                        menuViewModel.addMenu(form);
                    }
                    dialog.dismiss();
                }));

        dialog.show();
    }

    private MenuForm buildFormFromView(RadioGroup rgType, TextInputEditText etParentId,
                                       TextInputEditText etName, TextInputEditText etPath,
                                       TextInputEditText etComponent, TextInputEditText etPerm,
                                       TextInputEditText etIcon, TextInputEditText etRedirect,
                                       TextInputEditText etSort, RadioGroup rgVisible,
                                       String existingId) {
        MenuForm form = new MenuForm();
        if (existingId != null) {
            form.setId(existingId);
        }
        form.setType(typeFromRadio(rgType));
        String parentIdStr = getText(etParentId);
        form.setParentId(TextUtils.isEmpty(parentIdStr) ? 0 : Integer.parseInt(parentIdStr));
        form.setName(getText(etName));
        form.setPath(getText(etPath));
        form.setComponent(getText(etComponent));
        form.setPerm(getText(etPerm));
        form.setIcon(getText(etIcon));
        form.setRedirect(getText(etRedirect));
        String sortStr = getText(etSort);
        form.setSort(TextUtils.isEmpty(sortStr) ? 1 : Integer.parseInt(sortStr));
        form.setVisible(visibleFromRadio(rgVisible));
        return form;
    }

    private String getText(TextInputEditText et) {
        return et.getText() != null ? et.getText().toString().trim() : "";
    }

    private int typeFromRadio(RadioGroup rgType) {
        int checkedId = rgType.getCheckedRadioButtonId();
        if (checkedId == R.id.rb_type_dir) return 1;
        if (checkedId == R.id.rb_type_menu) return 2;
        if (checkedId == R.id.rb_type_button) return 3;
        return 1;
    }

    private int visibleFromRadio(RadioGroup rgVisible) {
        int checkedId = rgVisible.getCheckedRadioButtonId();
        return checkedId == R.id.rb_visible_yes ? 1 : 0;
    }

    private void selectTypeRadio(RadioGroup rgType, int type) {
        RadioButton rb;
        if (type == 2) {
            rb = rgType.findViewById(R.id.rb_type_menu);
        } else if (type == 3) {
            rb = rgType.findViewById(R.id.rb_type_button);
        } else {
            rb = rgType.findViewById(R.id.rb_type_dir);
        }
        rb.setChecked(true);
    }

    private void selectVisibleRadio(RadioGroup rgVisible, int visible) {
        RadioButton rb = rgVisible.findViewById(visible == 1 ? R.id.rb_visible_yes : R.id.rb_visible_no);
        rb.setChecked(true);
    }
}
