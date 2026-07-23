package com.pei.dehaze.ui.algorithm;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentAlgorithmBinding;
import com.pei.dehaze.ui.algorithm_select.AlgorithmSelectActivity;

public class AlgorithmFragment extends Fragment {

    private FragmentAlgorithmBinding binding;

    private final ActivityResultLauncher<Intent> algorithmSelectLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() != android.app.Activity.RESULT_OK || result.getData() == null) {
                    return;
                }
                Intent data = result.getData();
                long algorithmId = data.getLongExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_ID, 0L);
                if (algorithmId <= 0) return;
                Bundle args = new Bundle();
                args.putLong("algorithm_id", algorithmId);
                Navigation.findNavController(requireActivity(), R.id.nav_host_fragment_content_main)
                        .navigate(R.id.action_global_presentationActivity, args);
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentAlgorithmBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.btnAlgorithmManage.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), AlgorithmListActivity.class)));

        binding.btnAlgorithmSelect.setOnClickListener(v ->
                algorithmSelectLauncher.launch(new Intent(getActivity(), AlgorithmSelectActivity.class)));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
