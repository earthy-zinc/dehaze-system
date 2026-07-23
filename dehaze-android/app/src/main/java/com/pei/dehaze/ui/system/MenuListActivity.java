package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import com.pei.dehaze.utils.ToastUtils;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityMenuListBinding;
import com.pei.dehaze.ui.system.adapter.MenuAdapter;
import com.pei.dehaze.ui.system.viewmodel.MenuViewModel;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuType;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.utils.StringUtils;

import java.util.List;

public class MenuListActivity extends AppCompatActivity {

    private MenuViewModel menuViewModel;
    private MenuAdapter menuAdapter;
    private ActivityMenuListBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMenuListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        binding.toolbar.setNavigationOnClickListener(v -> finish());

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
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(menuAdapter);

        binding.swipeRefresh.setOnRefreshListener(() -> menuViewModel.loadMenus(null));

        binding.btnSearch.setOnClickListener(v -> {
            String keywords = StringUtils.getText(binding.etSearch);
            menuViewModel.loadMenus(keywords.isEmpty() ? null : keywords);
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etSearch.setText("");
            menuViewModel.loadMenus(null);
        });

        binding.btnAdd.setOnClickListener(v -> showFormDialog(null));
    }

    private void initViewModel() {
        menuViewModel = new ViewModelProvider(this).get(MenuViewModel.class);
    }

    private void setupObservers() {
        menuViewModel.getMenuList().observe(this, menus -> menuAdapter.setMenuTree(menus));

        menuViewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        menuViewModel.getError().observe(this, errorMsg -> {
            if (!TextUtils.isEmpty(errorMsg)) {
                ToastUtils.showShort(this, errorMsg);
            }
        });

        menuViewModel.getOperationResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                ToastUtils.showShort(this, result);
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
                    String name = StringUtils.getText(etName);
                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(this, "菜单名称不能为空");
                        return;
                    }
                    String sortStr = StringUtils.getText(etSort);
                    if (TextUtils.isEmpty(sortStr)) {
                        ToastUtils.showShort(this, "排序不能为空");
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
        String parentIdStr = StringUtils.getText(etParentId);
        form.setParentId(TextUtils.isEmpty(parentIdStr) ? 0 : Integer.parseInt(parentIdStr));
        form.setName(StringUtils.getText(etName));
        form.setPath(StringUtils.getText(etPath));
        form.setComponent(StringUtils.getText(etComponent));
        form.setPerm(StringUtils.getText(etPerm));
        form.setIcon(StringUtils.getText(etIcon));
        form.setRedirect(StringUtils.getText(etRedirect));
        String sortStr = StringUtils.getText(etSort);
        form.setSort(TextUtils.isEmpty(sortStr) ? 1 : Integer.parseInt(sortStr));
        form.setVisible(visibleFromRadio(rgVisible));
        return form;
    }

    private MenuType typeFromRadio(RadioGroup rgType) {
        int checkedId = rgType.getCheckedRadioButtonId();
        if (checkedId == R.id.rb_type_dir) return MenuType.CATALOG;
        if (checkedId == R.id.rb_type_menu) return MenuType.MENU;
        if (checkedId == R.id.rb_type_button) return MenuType.BUTTON;
        return MenuType.CATALOG;
    }

    private int visibleFromRadio(RadioGroup rgVisible) {
        int checkedId = rgVisible.getCheckedRadioButtonId();
        return checkedId == R.id.rb_visible_yes ? 1 : 0;
    }

    private void selectTypeRadio(RadioGroup rgType, MenuType type) {
        RadioButton rb;
        if (type == MenuType.MENU) {
            rb = rgType.findViewById(R.id.rb_type_menu);
        } else if (type == MenuType.BUTTON) {
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
