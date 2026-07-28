package com.pei.dehaze.service;

import com.pei.dehaze.model.entity.SysRating;

public interface LowRatingAlertService {

    boolean checkAndAlert(SysRating rating);

    boolean sendNormalAlert(SysRating rating);

    boolean sendUrgentAlert(Long algorithmId);

    boolean sendSevereAlert();
}
