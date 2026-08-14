/**
 * Error classes for DXMT JavaScript SDK.
 *
 * This module provides custom error classes for different types of failures.
 */

/**
 * Base error class for all DXMT errors.
 */
export class DXMTError extends Error {
  public readonly code?: string;
  public readonly details?: any;

  constructor(message: string, code?: string, details?: any) {
    super(message);
    this.name = 'DXMTError';
    this.code = code;
    this.details = details;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

/**
 * Error thrown when connection to DXMT server fails.
 */
export class DXMTConnectionError extends DXMTError {
  constructor(message: string, details?: any) {
    super(message, 'CONNECTION_ERROR', details);
    this.name = 'DXMTConnectionError';
  }
}

/**
 * Error thrown when table operations fail.
 */
export class DXMTTableError extends DXMTError {
  constructor(message: string, code?: string, details?: any) {
    super(message, code || 'TABLE_ERROR', details);
    this.name = 'DXMTTableError';
  }
}

/**
 * Error thrown when record operations fail.
 */
export class DXMTRecordError extends DXMTError {
  constructor(message: string, code?: string, details?: any) {
    super(message, code || 'RECORD_ERROR', details);
    this.name = 'DXMTRecordError';
  }
}

/**
 * Error thrown when search operations fail.
 */
export class DXMTSearchError extends DXMTError {
  constructor(message: string, code?: string, details?: any) {
    super(message, code || 'SEARCH_ERROR', details);
    this.name = 'DXMTSearchError';
  }
}

/**
 * Error thrown when validation fails.
 */
export class DXMTValidationError extends DXMTError {
  constructor(message: string, details?: any) {
    super(message, 'VALIDATION_ERROR', details);
    this.name = 'DXMTValidationError';
  }
}

/**
 * Error thrown when request times out.
 */
export class DXMTTimeoutError extends DXMTError {
  constructor(message: string, details?: any) {
    super(message, 'TIMEOUT_ERROR', details);
    this.name = 'DXMTTimeoutError';
  }
}

/**
 * Error thrown when authentication fails.
 */
export class DXMTAuthenticationError extends DXMTError {
  constructor(message: string, details?: any) {
    super(message, 'AUTHENTICATION_ERROR', details);
    this.name = 'DXMTAuthenticationError';
  }
}

/**
 * Parse error response from server and create appropriate error.
 */
export function parseErrorResponse(
  status: number,
  data: any
): DXMTError {
  const message = data?.message || data?.error || 'Unknown error';
  const code = data?.code;
  const details = data?.details;

  switch (status) {
    case 400:
      return new DXMTValidationError(message, details);
    case 401:
    case 403:
      return new DXMTAuthenticationError(message, details);
    case 404:
      return new DXMTTableError(`Not found: ${message}`, 'NOT_FOUND', details);
    case 408:
      return new DXMTTimeoutError(message, details);
    case 409:
      return new DXMTRecordError(
        `Conflict: ${message}`,
        'CONFLICT',
        details
      );
    case 500:
    case 502:
    case 503:
    case 504:
      return new DXMTConnectionError(
        `Server error: ${message}`,
        details
      );
    default:
      return new DXMTError(message, code, details);
  }
}
