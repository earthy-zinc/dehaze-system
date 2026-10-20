package com.pei.clarity.service;

import com.pei.clarity.model.entity.SysRating;

public interface LowRatingAlertService {

    boolean checkAndAlert(SysRating rating);

    boolean sendNormalAlert(SysRating rating);

    boolean sendUrgentAlert(String modelId);

    boolean sendSevereAlert();
}
