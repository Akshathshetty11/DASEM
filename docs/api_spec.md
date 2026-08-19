# REST API Specifications

Base URL: `/api/v1`

---

## 1. Analytics & Metrics

### `GET /analytics/summary`
Returns aggregated metrics for the dashboard overview.

**Response (200 OK)**:
```json
{
  "total_videos": 12,
  "total_accidents": 42,
  "severity_breakdown": {
    "Minor": 20,
    "Major": 15,
    "Critical": 7
  },
  "hazards": {
    "fire_incidents": 3,
    "smoke_incidents": 5
  },
  "recent_accidents": [...]
}
```

---

## 2. Accident History & Verification

### `GET /accidents`
Retrieve list of detected accident events with optional severity and status filtering.

**Query Parameters**:
- `severity` (optional): `Minor`, `Major`, `Critical`
- `status` (optional): `unverified`, `confirmed`, `false_positive`
- `page` (default: 1)
- `limit` (default: 20)

**Response (200 OK)**:
```json
{
  "accidents": [...],
  "total": 42,
  "page": 1,
  "pages": 3
}
```

### `GET /accidents/<id>`
Get detailed telemetry and involved vehicles for a single accident event.

### `PATCH /accidents/<id>`
Update operator verification status.

**Request Body**:
```json
{
  "status": "confirmed"
}
```

---

## 3. Video Upload & Status

### `POST /videos/upload`
Upload a raw traffic video file (`multipart/form-data`).

### `GET /videos/<id>/status`
Check processing progress and status.

---

## 4. Reports Export

### `POST /reports/export`
Generate PDF or CSV audit report.

**Request Body**:
```json
{
  "format": "pdf"
}
```
