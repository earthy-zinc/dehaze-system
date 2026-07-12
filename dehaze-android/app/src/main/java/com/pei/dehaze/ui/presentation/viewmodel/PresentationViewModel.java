package com.pei.dehaze.ui.presentation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.PresentationRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class PresentationViewModel extends ViewModel {

    private final PresentationRepository presentationRepository;

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> historyList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();

    private String originalImageUrl;

    public PresentationViewModel() {
        presentationRepository = new PresentationRepository();
    }

    public void uploadImage(File imageFile) {
        loading.setValue(true);
        presentationRepository.uploadImage(imageFile, new PresentationRepository.UploadCallback() {
            @Override
            public void onSuccess(FileInfo fileInfo) {
                uploadedFile.postValue(fileInfo);
                originalImageUrl = fileInfo.getUrl();
                operationResult.postValue("图片上传成功");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadAlgorithmOptions() {
        loading.setValue(true);
        presentationRepository.getAlgorithmOptions(new PresentationRepository.OptionsCallback() {
            @Override
            public void onSuccess(List<Option> options) {
                algorithmOptions.postValue(options);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadAlgorithms(AlgorithmQuery query) {
        loading.setValue(true);
        presentationRepository.getAlgorithmList(query, new PresentationRepository.AlgorithmListCallback() {
            @Override
            public void onSuccess(List<Algorithm> algorithms) {
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void getAlgorithmDetail(int id) {
        loading.setValue(true);
        presentationRepository.getAlgorithmDetail(id, new PresentationRepository.AlgorithmDetailCallback() {
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

    public void predict(long algorithmId, String params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        PredParam param = new PredParam();
        param.setAlgorithmId(algorithmId);
        param.setImageUrl(originalImageUrl);
        param.setParams(params);
        loading.setValue(true);
        presentationRepository.getPrediction(param, new PresentationRepository.PredictionCallback() {
            @Override
            public void onSuccess(PredResult result) {
                predictionResult.postValue(result);
                operationResult.postValue("去雾处理完成");
                loading.postValue(false);
                loadHistory();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadHistory() {
        presentationRepository.listPredictionLogs(1, 20, new PresentationRepository.PredictionLogListCallback() {
            @Override
            public void onSuccess(List<PredictionLogVO> logs) {
                historyList.postValue(logs != null ? logs : new ArrayList<>());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }

    public LiveData<Algorithm> getAlgorithmDetail() {
        return algorithmDetail;
    }

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<List<PredictionLogVO>> getHistoryList() {
        return historyList;
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

    public String getOriginalImageUrl() {
        return originalImageUrl;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }
}
