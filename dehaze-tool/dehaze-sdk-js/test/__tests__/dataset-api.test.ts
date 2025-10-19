import { DatasetAPI } from 'dehaze-sdk-js';

describe('DatasetAPI', () => {
  test('should have all required methods', () => {
    expect(typeof DatasetAPI.getList).toBe('function');
    expect(typeof DatasetAPI.getOptions).toBe('function');
    expect(typeof DatasetAPI.getDatasetInfoById).toBe('function');
    expect(typeof DatasetAPI.getImageItem).toBe('function');
    expect(typeof DatasetAPI.add).toBe('function');
    expect(typeof DatasetAPI.update).toBe('function');
    expect(typeof DatasetAPI.deleteByIds).toBe('function');
    expect(typeof DatasetAPI.addDatasetItem).toBe('function');
    expect(typeof DatasetAPI.updateDatasetItem).toBe('function');
    expect(typeof DatasetAPI.deleteDatasetItem).toBe('function');
    expect(typeof DatasetAPI.uploadItemImage).toBe('function');
    expect(typeof DatasetAPI.updateItemImage).toBe('function');
    expect(typeof DatasetAPI.deleteItemImage).toBe('function');
  });
});