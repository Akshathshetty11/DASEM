-- Dynamic Accident Severity & Multi-Object Telemetry
-- SQLite Database Schema Definition

-- Table: videos
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    processed_path VARCHAR(500),
    duration_seconds REAL DEFAULT 0.0,
    fps REAL DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'uploaded',
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    cars_count INTEGER DEFAULT 0,
    buses_count INTEGER DEFAULT 0,
    trucks_count INTEGER DEFAULT 0,
    bikes_count INTEGER DEFAULT 0,
    persons_count INTEGER DEFAULT 0,
    fire_detected BOOLEAN DEFAULT 0,
    smoke_detected BOOLEAN DEFAULT 0,
    detection_summary_json TEXT
);

-- Table: accidents
CREATE TABLE IF NOT EXISTS accidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    frame_number INTEGER NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    severity_score REAL NOT NULL,
    vehicle_count INTEGER DEFAULT 1,
    cars_count INTEGER DEFAULT 0,
    buses_count INTEGER DEFAULT 0,
    trucks_count INTEGER DEFAULT 0,
    bikes_count INTEGER DEFAULT 0,
    persons_count INTEGER DEFAULT 0,
    fire_detected BOOLEAN DEFAULT 0,
    smoke_detected BOOLEAN DEFAULT 0,
    snapshot_path VARCHAR(500),
    clip_path VARCHAR(500),
    bbox_json TEXT,
    verification_status VARCHAR(30) DEFAULT 'unverified',
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- Table: accident_vehicles
CREATE TABLE IF NOT EXISTS accident_vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accident_id INTEGER NOT NULL,
    track_id INTEGER,
    vehicle_type VARCHAR(50) DEFAULT 'car',
    pre_impact_speed REAL DEFAULT 0.0,
    post_impact_speed REAL DEFAULT 0.0,
    impact_force_index REAL DEFAULT 0.0,
    FOREIGN KEY (accident_id) REFERENCES accidents(id) ON DELETE CASCADE
);

-- Table: reports
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_name VARCHAR(255) NOT NULL,
    report_type VARCHAR(10) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_accidents_video_id ON accidents(video_id);
CREATE INDEX IF NOT EXISTS idx_accidents_severity ON accidents(severity_level);
CREATE INDEX IF NOT EXISTS idx_accidents_detected_at ON accidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_accident_vehicles_accident ON accident_vehicles(accident_id);
