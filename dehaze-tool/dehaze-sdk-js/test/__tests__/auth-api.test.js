"use strict";
Object.defineProperty(exports, "__esModule", {value: true});
const dehaze_sdk_js_1 = require("dehaze-sdk-js");
// Mock data
const mockLoginData = {
    username: 'testuser',
    password: 'testpass'
};
const mockLoginResult = {
    accessToken: 'mock-access-token',
    expires: 3600000,
    refreshToken: 'mock-refresh-token',
    tokenType: 'Bearer'
};
const mockCaptchaResult = {
    captchaKey: 'mock-captcha-key',
    captchaBase64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='
};
describe('AuthAPI', () => {
    it('should login with credentials', async () => {
        // Mock the HTTP request
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '登录成功',
                data: mockLoginResult
            })
        });
        const loginResult = await dehaze_sdk_js_1.AuthAPI.login(mockLoginData);
        expect(loginResult).toEqual(mockLoginResult);
    });
    it('should logout', async () => {
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '注销成功',
                data: null
            })
        });
        const result = await dehaze_sdk_js_1.AuthAPI.logout();
        expect(result.code).toBe('00000');
        expect(result.msg).toBe('注销成功');
    });
    it('should get captcha', async () => {
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '操作成功',
                data: mockCaptchaResult
            })
        });
        const captchaResult = await dehaze_sdk_js_1.AuthAPI.getCaptcha();
        expect(captchaResult).toEqual(mockCaptchaResult);
    });
});
//# sourceMappingURL=auth-api.test.js.map
