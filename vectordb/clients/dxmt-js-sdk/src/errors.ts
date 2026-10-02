/**
 * Error classes for DXMT JavaScript SDK.
 *
 * This module provides custom error classes for different types of failures.
 */

/**
 * Base error class for all DXMT errors.
 */
export class DXMTError extends Error {
  /** The server's error code when it sent one (`TABLE_NOT_FOUND`, `DIMENSION_MISMATCH`,
   *  `FEATURE_DISABLED`, ...), otherwise this client's own. */
  public code?: string;
  public readonly details?: any;
  /** The HTTP status, when the error came from a response. */
  public status?: number;

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
  // Standard Rust responses use { success: false, error: { code, message, details } };
  // compatibility handlers may still return a flat { error, details } object.
  const nested = data?.error && typeof data.error === 'object' ? data.error : data;
  const message = nested?.message || (typeof data?.error === 'string' ? data.error : undefined) || 'Unknown error';
  const code = nested?.code || data?.code;
  const details = nested?.details || data?.details;

  // The class is chosen by status so callers can `instanceof`; the *code* is the server's
  // own whenever it sent one. These used to be overwritten by each class's constant --
  // `DIMENSION_MISMATCH` arrived as `VALIDATION_ERROR`, `TABLE_NOT_FOUND` as `NOT_FOUND`,
  // `FEATURE_DISABLED` on a 503 as `CONNECTION_ERROR` -- so a caller could tell *that* the
  // server refused but not *why*, which is the only question an error code answers.
  const err = classify(status, message, code, details);
  if (code) err.code = code;
  err.status = status;
  return err;
}

function classify(status: number, message: string, code: string | undefined, details: any): DXMTError {
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
    case 429:
      return new DXMTError(message, code || 'TOO_MANY_REQUESTS', details);
    case 501:
      return new DXMTError(message, code || 'NOT_IMPLEMENTED', details);
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
