import {AuthAPI, CaptchaResult, LoginData} from 'dehaze-sdk-js';

// Mock data
const mockLoginData: LoginData = {
    username: 'testuser',
    password: 'testpass'
};

const mockLoginResult = {
    accessToken: 'mock-access-token',
    expires: 3600000,
    refreshToken: 'mock-refresh-token',
    tokenType: 'Bearer'
};

const mockCaptchaResult: CaptchaResult = {
    captchaKey: 'mock-captcha-key',
    captchaBase64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='
};

// 简单的测试，只验证类型是否正确
describe('AuthAPI Types', () => {
    it('should have correct type for LoginData', () => {
        const loginData: LoginData = mockLoginData;
        expect(loginData.username).toBe('testuser');
        expect(loginData.password).toBe('testpass');
    });

    it('should have correct type for CaptchaResult', () => {
        const captchaResult: CaptchaResult = mockCaptchaResult;
        expect(captchaResult.captchaKey).toBe('mock-captcha-key');
    });

    it('should have all required methods', () => {
        expect(typeof AuthAPI.login).toBe('function');
        expect(typeof AuthAPI.logout).toBe('function');
        expect(typeof AuthAPI.getCaptcha).toBe('function');
    });
});
