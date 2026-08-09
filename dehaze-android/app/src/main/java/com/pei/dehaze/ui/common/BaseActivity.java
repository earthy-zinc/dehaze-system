package com.pei.dehaze.ui.common;

import android.view.MenuItem;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.pei.dehaze.utils.ToastUtils;

/**
 * Activity 基类，收敛 toolbar / error / operationResult 等重复样板。
 *
 * <p>子类按需调用：
 * <ul>
 *   <li>{@link #setupToolbar(Toolbar, String)} 或 {@link #setupActionBar(String)} 统一 toolbar 与返回键</li>
 *   <li>{@link #observeError(BaseViewModel)} 统一错误提示，消除各 Activity 重复 observe + Toast 样板</li>
 *   <li>{@link #observeOperationResult(BaseViewModel, Runnable)} 统一操作成功提示与后续动作</li>
 * </ul>
 *
 * <p>不强制注入 ViewModel 泛型：Activity 类型多样（有 VM / 无 VM / 多 VM），泛型约束反而增加复杂度。
 * 沉浸页（Compare/Evaluation/Presentation）若不需要 toolbar，可不调用 setupToolbar。
 */
public abstract class BaseActivity extends AppCompatActivity {

    /**
     * 使用布局内自定义 Toolbar 作为 ActionBar，设置标题与返回键。
     */
    protected void setupToolbar(@NonNull Toolbar toolbar, String title) {
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            if (title != null) {
                getSupportActionBar().setTitle(title);
            }
        }
        toolbar.setNavigationOnClickListener(v -> finish());
    }

    /**
     * 使用默认 ActionBar，设置标题与返回键。
     */
    protected void setupActionBar(String title) {
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            if (title != null) {
                getSupportActionBar().setTitle(title);
            }
        }
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    /**
     * 统一观察 ViewModel 的 error，弹出 Toast 后清除，避免旋转屏重复弹出旧错误。
     */
    protected void observeError(@NonNull BaseViewModel vm) {
        vm.getError().observe(this, msg -> {
            if (msg != null && !msg.isEmpty()) {
                ToastUtils.showShort(this, msg);
                vm.clearError();
            }
        });
    }

    /**
     * 统一观察 ViewModel 的 operationResult，弹出 Toast 后清除，并执行后续动作（如刷新列表、finish）。
     *
     * @param onSuccess 操作成功提示后的后续动作，可为 null
     */
    protected void observeOperationResult(@NonNull BaseViewModel vm, Runnable onSuccess) {
        vm.getOperationResult().observe(this, msg -> {
            if (msg != null && !msg.isEmpty()) {
                ToastUtils.showShort(this, msg);
                vm.clearOperationResult();
                if (onSuccess != null) {
                    onSuccess.run();
                }
            }
        });
    }
}
