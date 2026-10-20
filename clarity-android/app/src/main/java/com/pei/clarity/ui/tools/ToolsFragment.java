package com.pei.clarity.ui.tools;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.clarity.R;
import com.pei.clarity.databinding.FragmentToolsBinding;
import com.pei.clarity.ui.batch.BatchActivity;
import com.pei.clarity.ui.input.InputHistoryActivity;
import com.pei.clarity.ui.metrics_manage.MetricsManageActivity;
import com.pei.clarity.ui.personal.CreditsActivity;
import com.pei.clarity.ui.personal.FavoritesActivity;

public class ToolsFragment extends Fragment {

    private ToolsViewModel toolsViewModel;
    private FragmentToolsBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentToolsBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        toolsViewModel = new ViewModelProvider(this).get(ToolsViewModel.class);

        initViews();
        setupObservers();
    }

    private void initViews() {
        QuickEntryAdapter quickAdapter = new QuickEntryAdapter(entry -> {
            switch (entry.getId()) {
                case 1:
                    open(InputHistoryActivity.class);
                    break;
                case 2:
                    open(FavoritesActivity.class);
                    break;
                case 3:
                    open(BatchActivity.class);
                    break;
                case 4:
                    open(CreditsActivity.class);
                    break;
                default:
                    break;
            }
        });
        binding.rvQuickEntries.setLayoutManager(
                new LinearLayoutManager(requireContext(), RecyclerView.HORIZONTAL, false));
        binding.rvQuickEntries.setAdapter(quickAdapter);

        FeatureGridAdapter featureAdapter = new FeatureGridAdapter(item -> {
            switch (item.getAction()) {
                case "upload":
                    open(InputHistoryActivity.class);
                    break;
                case "dehaze":
                    navigateTo(R.id.dehazeFragment);
                    break;
                case "batch":
                    open(BatchActivity.class);
                    break;
                case "dataset":
                    navigateTo(R.id.datasetFragment);
                    break;
                case "evaluation":
                    navigateTo(R.id.action_global_evaluationActivity);
                    break;
                case "metrics":
                    open(MetricsManageActivity.class);
                    break;
                case "credits":
                    open(CreditsActivity.class);
                    break;
                default:
                    break;
            }
        });
        binding.rvFeatureGrid.setLayoutManager(new GridLayoutManager(requireContext(), 3));
        binding.rvFeatureGrid.setAdapter(featureAdapter);
    }

    private void open(Class<?> activityClass) {
        startActivity(new Intent(getActivity(), activityClass));
    }

    private void navigateTo(int destinationId) {
        NavController navController = Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
        navController.navigate(destinationId);
    }

    private void setupObservers() {
        toolsViewModel.getQuickEntries().observe(getViewLifecycleOwner(), entries -> {
            RecyclerView.Adapter<?> adapter = binding.rvQuickEntries.getAdapter();
            if (adapter instanceof QuickEntryAdapter) {
                ((QuickEntryAdapter) adapter).submitList(entries);
            }
        });

        toolsViewModel.getFeatureItems().observe(getViewLifecycleOwner(), items -> {
            RecyclerView.Adapter<?> adapter = binding.rvFeatureGrid.getAdapter();
            if (adapter instanceof FeatureGridAdapter) {
                ((FeatureGridAdapter) adapter).submitList(items);
            }
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
