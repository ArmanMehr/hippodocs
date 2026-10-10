# Step 3: Documenting the Error Handling Path

## Problem

The HTTP middleware in `app/main.py` currently owns several responsibilities at
once:

- request logging,
- translating database errors into `503` responses,
- handling every other unexpected error as a generic `500`.

This overlaps with the dedicated exception handlers already registered for
`AppError` and `RequestValidationError`. A single failing request can therefore
be logged more than once, by both the middleware and a dedicated handler. That
duplication makes the behavior hard to reason about and hard to change safely.

## Goal of this step

Do **not** rewrite the code yet. First pin down how errors behave **today** with
integration tests. The tests describe, for each kind of error:

- which HTTP status and body the client receives,
- which log event fires, at which level, and how many times.

Only after that baseline exists should you decide which responsibility to move
out of the middleware and into a dedicated exception handler. The baseline turns
that refactor into a test-verified change instead of a guess.

## Key architecture fact

Starlette assembles the request pipeline in this order (outermost first):

```
ServerErrorMiddleware
  -> your @app.middleware("http")        # request_logging_middleware
    -> ExceptionMiddleware               # holds @app.exception_handler(...)
      -> router
```

Two consequences follow from this ordering:

1. Your `request_logging_middleware` is **outer** to the exception handlers.
   An `AppError` raised inside a route is converted to a JSON response by
   `app_error_handler` **before** the middleware ever sees it. The middleware
   then logs the result as an ordinary `http_request_completed` event. So an
   `AppError` produces **two** log records: one `application_error` warning from
   the handler, and one `http_request_completed` info from the middleware. This
   double logging is exactly the overlap this step is about.

2. Only exceptions that have **no** registered handler reach the middleware's
   `except` blocks. That is why raw `OperationalError` / `InterfaceError` become
   `503`, and every other unhandled `Exception` becomes `500`, both produced by
   the middleware itself.

## Where the tests live

Put them in a single integration test file, for example
`tests/e2e/test_error_handling.py`. These behaviors only exist at the level of
the full ASGI app (middleware plus exception handlers), so a unit test cannot
cover them.

## Test setup

Reuse the pattern already used in `tests/e2e/test_workspaces_api.py`:

- Build a `TestClient(app)` with `app.dependency_overrides` for the services you
  want to control.
- Keep the Langfuse-disabling environment lines at the top of the module, same
  as the existing e2e test, so observability stays off during the run.

To assert logs, do not fight structlog or `caplog`. Patch the methods on
`app.main.logger` directly, in the same style already used against
`observability.logger`:

```python
info_mock = mocker.patch.object(app.main.logger, "info")
warning_mock = mocker.patch.object(app.main.logger, "warning")
exception_mock = mocker.patch.object(app.main.logger, "exception")
```

Then inspect `call_args` / `call_count` and the `event` keyword.

## Scenarios

| Scenario | Expected result |
| --- | --- |
| Successful request | `2xx` and a log with status code and duration |
| Validation error | `422` with an `ErrorPayload` body |
| Request to a missing workspace | `404` with the right `error_code` |
| Database error (mocked) | `503` with no internal details leaked |
| Unexpected error (mocked) | Generic `500`, traceback in the server log |

### 1. Successful request

- Request: `GET /health`.
- Expect `200`.
- Expect an `X-Request-ID` response header.
- Expect `logger.info` called with `event="http_request_completed"`, carrying
  `status_code` and `duration_ms`.

### 2. Validation error

- Request: `POST /v1/workspaces/` with an invalid body (for example `{}` or a
  wrong field type).
- Expect `422`.
- Expect a body whose keys match `ErrorPayload` exactly: `detail`, `error_code`,
  `timestamp`, with `error_code == "request_validation_error"`.
- Expect the handler warning `application_error` to fire.

### 3. Missing workspace

- Request: `GET /v1/workspaces/{id}` for an id the fake unit of work does not
  know.
- Expect `404` with `error_code == "workspace_not_found"`.
- This is the double-log case. Assert **both**:
  - `application_error` warning from `app_error_handler`,
  - `http_request_completed` info from the middleware.
- Documenting this pair is the point: it is the clearest candidate for cleanup
  in the later refactor.

### 4. Database error (mocked)

- Make a dependency return a service whose call raises
  `sqlalchemy.exc.OperationalError` (constructed the way the repository already
  imports it).
- Expect `503` with `error_code == "database_unavailable"`.
- Critically, assert that the internal `str(exc.orig)` does **not** appear in
  `response.json()`. It should appear only in the `database_unavailable` server
  log.
- Expect the middleware warning `database_unavailable` to fire.

### 5. Unexpected error (mocked)

- Make a dependency raise a plain `RuntimeError`.
- Expect `500` with `error_code == "internal_error"` and
  `detail == "Internal server error"`.
- Assert no traceback appears in the response body.
- Expect `logger.exception` called with `event="http_request_failed"`, so the
  traceback lands in the server log rather than the response.

## What this gives you

Once these tests are green, the repository history documents the current
behavior, so the follow-up refactor (removing the database and generic
`except Exception` branches from the middleware and moving them into dedicated
exception handlers) becomes a safe, test-verified change.

The tests that assert **both** an `application_error` and a
`http_request_completed` record are the ones that will legitimately need to
change after the refactor. They are your signal for where responsibility still
overlaps.
