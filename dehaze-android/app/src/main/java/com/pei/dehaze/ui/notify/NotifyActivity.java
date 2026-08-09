package com.pei.dehaze.ui.notify;

import android.os.Bundle;
import android.view.View;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.databinding.ActivityNotifyBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.NotificationSettingAPI;
import com.pei.dehaze.sdk.model.message.NotificationSettings;
import com.pei.dehaze.sdk.model.message.NotificationSettingsForm;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 消息设置 — 通知开关 + 免打扰时段
 */
public class NotifyActivity extends BaseActivity {

    private ActivityNotifyBinding binding;
    private NotifyViewModel viewModel;

    private boolean pushEnabled = true;
    private boolean dndEnabled = false;
    private String dndStart = "22:00";
    private String dndEnd = "08:00";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityNotifyBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("消息设置");

        viewModel = new ViewModelProvider(this).get(NotifyViewModel.class);
        viewModel.getSettings().observe(this, this::applySettings);
        observeError(viewModel);

        setupSwitches();
        setupDnd();
        viewModel.loadSettings();
    }

    private void applySettings(NotificationSettings settings) {
        if (settings == null) return;
        if (settings.getPushEnabled() != null) {
            pushEnabled = settings.getPushEnabled();
            binding.swPush.setChecked(pushEnabled);
        }
        if (settings.getDndEnabled() != null) {
            dndEnabled = settings.getDndEnabled();
            binding.swDnd.setChecked(dndEnabled);
            binding.layoutDndTimes.setVisibility(dndEnabled ? View.VISIBLE : View.GONE);
        }
        if (settings.getDndStart() != null) {
            dndStart = settings.getDndStart();
            binding.tvDndStart.setText(dndStart);
        }
        if (settings.getDndEnd() != null) {
            dndEnd = settings.getDndEnd();
            binding.tvDndEnd.setText(dndEnd);
        }
    }

    private void setupSwitches() {
        binding.swPush.setOnCheckedChangeListener((btn, checked) -> {
            pushEnabled = checked;
            saveSettings();
            showToast("推送通知已" + (checked ? "开启" : "关闭"));
        });

        binding.swTaskComplete.setOnCheckedChangeListener((btn, checked) -> {
            saveSettings();
            showToast("处理完成提醒已" + (checked ? "开启" : "关闭"));
        });

        binding.swActivity.setOnCheckedChangeListener((btn, checked) -> {
            saveSettings();
            showToast("活动通知已" + (checked ? "开启" : "关闭"));
        });
    }

    private void setupDnd() {
        binding.swDnd.setOnCheckedChangeListener((btn, checked) -> {
            dndEnabled = checked;
            binding.layoutDndTimes.setVisibility(checked ? View.VISIBLE : View.GONE);
            saveSettings();
            showToast("免打扰模式已" + (checked ? "开启" : "关闭"));
        });

        binding.tvDndStart.setOnClickListener(v ->
                ToastUtils.showShort(this, "免打扰开始时间: " + dndStart + "（时间选择器将在后续版本支持）"));

        binding.tvDndEnd.setOnClickListener(v ->
                ToastUtils.showShort(this, "免打扰结束时间: " + dndEnd + "（时间选择器将在后续版本支持）"));
    }

    private void saveSettings() {
        NotificationSettingsForm form = new NotificationSettingsForm();
        form.setPushEnabled(pushEnabled);
        form.setDndEnabled(dndEnabled);
        form.setDndStart(dndStart);
        form.setDndEnd(dndEnd);
        viewModel.saveSettings(form);
    }

    private void showToast(String msg) {
        ToastUtils.showShort(this, msg);
    }

    public static class NotifyViewModel extends BaseViewModel {
        private final MutableLiveData<NotificationSettings> settings = new MutableLiveData<>();

        public LiveData<NotificationSettings> getSettings() {
            return settings;
        }

        public void loadSettings() {
            NotificationSettingAPI.get(RepositoryAdapters.wrap(withLoading(data -> settings.postValue(data))));
        }

        public void saveSettings(NotificationSettingsForm form) {
            NotificationSettingAPI.update(form, RepositoryAdapters.wrap(withLoading(v -> {
            })));
        }
    }
}
