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
import com.pei.dehaze.ui.system.DeptListActivity;
import com.pei.dehaze.ui.system.RoleListActivity;
import com.pei.dehaze.ui.system.UserListActivity;

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
        
        // 设置按钮点击事件
        binding.userManagementButton.setOnClickListener(v -> {
            Intent intent = new Intent(getActivity(), UserListActivity.class);
            startActivity(intent);
        });
        
        binding.roleManagementButton.setOnClickListener(v -> {
            Intent intent = new Intent(getActivity(), RoleListActivity.class);
            startActivity(intent);
        });
        
        binding.departmentManagementButton.setOnClickListener(v -> {
            Intent intent = new Intent(getActivity(), DeptListActivity.class);
            startActivity(intent);
        });
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}