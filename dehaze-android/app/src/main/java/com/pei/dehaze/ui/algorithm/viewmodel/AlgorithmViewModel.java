package com.pei.dehaze.ui.algorithm.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.AlgorithmRepository;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;

import java.util.List;

public class AlgorithmViewModel extends ViewModel {

    private final AlgorithmRepository algorithmRepository;

    private final MutableLiveData<List<Algorithm>> algorithmList = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();

    public AlgorithmViewModel() {
        algorithmRepository = new AlgorithmRepository();
    }

    public void loadAlgorithms(AlgorithmQuery query) {
        loading.setValue(true);
        algorithmRepository.getAlgorithms(query, new AlgorithmRepository.AlgorithmCallback() {
            @Override
            public void onSuccess(List<Algorithm> algorithms) {
                algorithmList.postValue(algorithms);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadAlgorithmDetail(int id) {
        loading.setValue(true);
        algorithmRepository.getAlgorithmDetail(id, new AlgorithmRepository.AlgorithmDetailCallback() {
            @Override
            public void onSuccess(Algorithm algorithm) {
                algorithmDetail.postValue(algorithm);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public LiveData<List<Algorithm>> getAlgorithmList() {
        return algorithmList;
    }

    public LiveData<Algorithm> getAlgorithmDetail() {
        return algorithmDetail;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }
}