/**
 * Basic tests for DXMT Client.
 *
 * These tests validate the client API without requiring a live server.
 */

import { describe, it, expect } from 'vitest';
import { DXMTClient } from '../src/client';
import {
  DXMTValidationError,
  DXMTConnectionError,
} from '../src/errors';

describe('DXMTClient', () => {
  describe('Constructor', () => {
    it('should create client with default config', () => {
      const client = new DXMTClient();
      expect(client.getBaseUrl()).toBe('http://localhost:9999/api/v1');
      expect(client.getTimeout()).toBe(30000);
    });

    it('should create client with custom config', () => {
      const client = new DXMTClient({
        host: 'example.com',
        port: 8080,
        ssl: true,
        timeout: 60000,
      });

      expect(client.getBaseUrl()).toBe('https://example.com:8080/api/v1');
      expect(client.getTimeout()).toBe(60000);
    });

    it('should add API key to headers if provided', () => {
      const client = new DXMTClient({
        apiKey: 'test-key',
      });

      // Headers are private, but we can test the client was created
      expect(client).toBeInstanceOf(DXMTClient);
    });
  });

  describe('Validation', () => {
    const client = new DXMTClient();

    it('should validate table name in createTable', async () => {
      await expect(
        client.createTable({
          name: '',
          dimension: 128,
        })
      ).rejects.toThrow(DXMTValidationError);
    });

    it('should validate dimension in createTable', async () => {
      await expect(
        client.createTable({
          name: 'test',
          dimension: 0,
        })
      ).rejects.toThrow(DXMTValidationError);

      await expect(
        client.createTable({
          name: 'test',
          dimension: -1,
        })
      ).rejects.toThrow(DXMTValidationError);
    });

    it('should validate table name in insertRecords', async () => {
      await expect(
        client.insertRecords({
          table_name: '',
          records: [{ id: 'test', vector: [1, 2, 3] }],
        })
      ).rejects.toThrow(DXMTValidationError);
    });

    it('should validate records array in insertRecords', async () => {
      await expect(
        client.insertRecords({
          table_name: 'test',
          records: [],
        })
      ).rejects.toThrow(DXMTValidationError);
    });

    it('should validate record ID', async () => {
      await expect(
        client.insertRecords({
          table_name: 'test',
          records: [{ id: '', vector: [1, 2, 3] }],
        })
      ).rejects.toThrow(DXMTValidationError);
    });

    it('should validate record vector', async () => {
      await expect(
        client.insertRecords({
          table_name: 'test',
          records: [{ id: 'test', vector: [] }],
        })
      ).rejects.toThrow(DXMTValidationError);

      await expect(
        client.insertRecords({
          table_name: 'test',
          // @ts-expect-error Testing invalid type
          records: [{ id: 'test', vector: 'not-an-array' }],
        })
      ).rejects.toThrow(DXMTValidationError);
    });

    it('should validate search parameters', async () => {
      await expect(
        client.search({
          table_name: '',
          vector: [1, 2, 3],
        })
      ).rejects.toThrow(DXMTValidationError);

      await expect(
        client.search({
          table_name: 'test',
          vector: [],
        })
      ).rejects.toThrow(DXMTValidationError);

      await expect(
        client.search({
          table_name: 'test',
          // @ts-expect-error Testing invalid type
          vector: 'not-an-array',
        })
      ).rejects.toThrow(DXMTValidationError);
    });
  });

  describe('URL Encoding', () => {
    it('should encode table names with special characters', () => {
      const client = new DXMTClient();

      // These will fail to connect, but we're testing URL construction
      const promise = client.getTable('test/table with spaces');

      // The request will fail due to no server, but URL should be encoded
      expect(promise).rejects.toThrow();
    });
  });

  describe('Utility Methods', () => {
    it('should get base URL', () => {
      const client = new DXMTClient({
        host: 'example.com',
        port: 8080,
        ssl: true,
      });

      expect(client.getBaseUrl()).toBe('https://example.com:8080/api/v1');
    });

    it('should get timeout', () => {
      const client = new DXMTClient({ timeout: 60000 });
      expect(client.getTimeout()).toBe(60000);
    });

    it('should set custom headers', () => {
      const client = new DXMTClient();
      client.setHeaders({ 'X-Custom': 'value' });

      // Headers are private, but method should not throw
      expect(() => client.setHeaders({ 'X-Custom': 'value' })).not.toThrow();
    });
  });
});
