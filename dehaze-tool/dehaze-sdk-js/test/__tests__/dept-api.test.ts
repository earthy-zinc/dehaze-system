import { DeptAPI } from 'dehaze-sdk-js';

describe('DeptAPI', () => {
  test('should have all required methods', () => {
    expect(typeof DeptAPI.getList).toBe('function');
    expect(typeof DeptAPI.getOptions).toBe('function');
    expect(typeof DeptAPI.getFormData).toBe('function');
    expect(typeof DeptAPI.add).toBe('function');
    expect(typeof DeptAPI.update).toBe('function');
    expect(typeof DeptAPI.deleteByIds).toBe('function');
  });
});