export class LoginData {
  username: string = '';
  password: string = '';
  captchaKey?: string = undefined;
  captchaCode?: string = undefined;
}

export class RegisterData {
  username: string = '';
  password: string = '';
  nickname: string = '';
  captchaKey?: string = undefined;
  captchaCode?: string = undefined;
}

export class LoginUser {
  id: number = 0;
  username: string = '';
  nickname: string = '';
}

export class LoginResult {
  sessionId: string = '';
  user?: LoginUser = undefined;
}

export class CaptchaResult {
  captchaKey: string = '';
  captchaBase64: string = '';
}

export class AuthUserInfo {
  userId: number = 0;
  username: string = '';
  nickname: string = '';
  avatar?: string = undefined;
  roles: string[] = [];
  perms: string[] = [];
}
