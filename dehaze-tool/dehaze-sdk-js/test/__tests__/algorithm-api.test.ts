import { AlgorithmAPI } from 'dehaze-sdk-js';

describe('AlgorithmAPI', () => {
  test('should have all required methods', () => {
    expect(typeof AlgorithmAPI.getList).toBe('function');
    expect(typeof AlgorithmAPI.getOption).toBe('function');
    expect(typeof AlgorithmAPI.getAlgorithmInfoById).toBe('function');
    expect(typeof AlgorithmAPI.add).toBe('function');
    expect(typeof AlgorithmAPI.update).toBe('function');
    expect(typeof AlgorithmAPI.deleteByIds).toBe('function');
  });
});