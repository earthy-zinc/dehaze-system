package com.pei.dehaze.ui.system;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.pei.dehaze.databinding.FragmentSystemManagementBinding;
import com.pei.dehaze.ui.file.FileListActivity;
import com.pei.dehaze.ui.input.InputHistoryActivity;
import com.pei.dehaze.ui.task.TaskListActivity;

public class SystemManagementFragment extends Fragment {

    private FragmentSystemManagementBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentSystemManagementBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // 组织与权限
        binding.userManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), UserListActivity.class)));

        binding.roleManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), RoleListActivity.class)));

        binding.departmentManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), DeptListActivity.class)));

        // 系统配置
        binding.menuManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), MenuListActivity.class)));

        binding.dictTypeManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), DictTypeListActivity.class)));

        binding.fileManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), FileListActivity.class)));

        // 任务与历史
        binding.taskManagementButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), TaskListActivity.class)));

        binding.inputHistoryButton.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), InputHistoryActivity.class)));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
