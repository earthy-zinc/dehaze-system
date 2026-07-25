package com.pei.dehaze.security;

import android.content.Context;
import android.content.SharedPreferences;

import com.pei.dehaze.sdk.utils.TokenStorage;

public class SharedPreferencesTokenStorage implements TokenStorage {

    private static final String PREF_NAME = "dehaze_auth_prefs";
    private static final String KEY_SESSION_ID = "session_id";

    private final SharedPreferences prefs;

    public SharedPreferencesTokenStorage(Context context) {
        this.prefs = context.getApplicationContext()
                .getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }

    @Override
    public void saveSessionId(String sessionId) {
        prefs.edit().putString(KEY_SESSION_ID, sessionId).apply();
    }

    @Override
    public String loadSessionId() {
        return prefs.getString(KEY_SESSION_ID, null);
    }

    @Override
    public void clearSessionId() {
        prefs.edit().remove(KEY_SESSION_ID).apply();
    }
}
