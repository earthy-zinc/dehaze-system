package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.Map;

@Data
public class NotificationSettings {
    private Boolean pushEnabled;
    private Boolean dndEnabled;
    private String dndStart;
    private String dndEnd;
    private Preferences preferences;

    @Data
    public static class Preferences {
        private Map<String, Map<String, Boolean>> typeChannels;
        private Map<String, Boolean> moduleSwitches;
    }
}
