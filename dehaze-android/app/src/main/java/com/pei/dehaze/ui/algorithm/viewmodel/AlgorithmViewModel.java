package com.pei.dehaze.ui.algorithm.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.AlgorithmRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmFavorite;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmViewModel extends ViewModel {

    private final AlgorithmRepository algorithmRepository;

    private final MutableLiveData<List<Algorithm>> algorithmList = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<List<Algorithm>> compareResult = new MutableLiveData<>();
    private final MutableLiveData<List<AlgorithmFavorite>> favoriteList = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();

    private String keywords = "";

    public AlgorithmViewModel() {
        algorithmRepository = new AlgorithmRepository();
    }

    public void loadAlgorithms() {
        loading.setValue(true);
        AlgorithmQuery query = new AlgorithmQuery();
        query.setKeywords(keywords);
        algorithmRepository.getAlgorithms(query, new AlgorithmRepository.Callback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
                algorithmList.postValue(data != null ? data : new ArrayList<>());
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
        algorithmRepository.getAlgorithmDetail(id, new AlgorithmRepository.Callback<Algorithm>() {
            @Override
            public void onSuccess(Algorithm data) {
                algorithmDetail.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addAlgorithm(Algorithm data) {
        loading.setValue(true);
        algorithmRepository.addAlgorithm(data, new AlgorithmRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("新增算法成功");
                loading.postValue(false);
                loadAlgorithms();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateAlgorithm(int id, Algorithm data) {
        loading.setValue(true);
        algorithmRepository.updateAlgorithm(id, data, new AlgorithmRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("修改算法成功");
                loading.postValue(false);
                loadAlgorithms();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteAlgorithms(String ids) {
        loading.setValue(true);
        algorithmRepository.deleteAlgorithms(ids, new AlgorithmRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除算法成功");
                loading.postValue(false);
                loadAlgorithms();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateAlgorithmStatus(long id, int status) {
        loading.setValue(true);
        algorithmRepository.updateAlgorithmStatus(id, status, new AlgorithmRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("状态更新成功");
                loading.postValue(false);
                loadAlgorithms();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void compareAlgorithms(String ids) {
        loading.setValue(true);
        algorithmRepository.compare(ids, new AlgorithmRepository.Callback<List<Algorithm>>() {
            @Override
            public void onSuccess(List<Algorithm> data) {
                compareResult.postValue(data != null ? data : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadFavorites() {
        loading.setValue(true);
        algorithmRepository.listFavorites(new AlgorithmRepository.Callback<List<AlgorithmFavorite>>() {
            @Override
            public void onSuccess(List<AlgorithmFavorite> data) {
                favoriteList.postValue(data != null ? data : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void toggleFavorite(long id) {
        algorithmRepository.toggleFavorite(id, new AlgorithmRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("收藏状态已更新");
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void loadOptions() {
        algorithmRepository.getOptions(new AlgorithmRepository.Callback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                algorithmOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords == null ? "" : keywords;
    }

    public void resetQuery() {
        this.keywords = "";
    }

    public LiveData<List<Algorithm>> getAlgorithmList() {
        return algorithmList;
    }

    public LiveData<Algorithm> getAlgorithmDetail() {
        return algorithmDetail;
    }

    public LiveData<List<Algorithm>> getCompareResult() {
        return compareResult;
    }

    public LiveData<List<AlgorithmFavorite>> getFavoriteList() {
        return favoriteList;
    }

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getOperationResult() {
        return operationResult;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearCompareResult() {
        compareResult.setValue(null);
    }
}
