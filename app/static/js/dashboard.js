// Dashboard Client Logic - Vehicle & Hazard Telemetry
let objectChartInstance = null;
let severityChartInstance = null;

function loadDashboardMetrics() {
    fetch('/api/v1/analytics/summary')
        .then(response => response.json())
        .then(data => {
            const objs = data.object_counts || {};
            document.getElementById('val-total-cars').innerText = objs.cars || 0;
            document.getElementById('val-total-buses').innerText = objs.buses || 0;
            document.getElementById('val-total-trucks').innerText = objs.trucks || 0;
            document.getElementById('val-total-bikes').innerText = objs.bikes || 0;
            document.getElementById('val-critical-accidents').innerText = data.severity_breakdown.Critical || 0;

            renderObjectDistributionChart(objs);
            renderSeverityChart(data.severity_breakdown);
            renderRecentAccidentsTable(data.recent_accidents);
        })
        .catch(error => console.error('Error loading metrics:', error));
}

function renderObjectDistributionChart(objs) {
    const ctx = document.getElementById('chart-object-distribution');
    if (!ctx) return;

    if (objectChartInstance) {
        objectChartInstance.destroy();
    }

    objectChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Cars (Blue)', 'Buses (Green)', 'Trucks (Orange)', 'Motorcycles (Yellow)'],
            datasets: [{
                label: 'Total Detected Count',
                data: [objs.cars || 0, objs.buses || 0, objs.trucks || 0, objs.bikes || 0],
                backgroundColor: [
                    'rgba(0, 128, 255, 0.85)',
                    'rgba(0, 255, 0, 0.85)',
                    'rgba(255, 140, 0, 0.85)',
                    'rgba(255, 255, 0, 0.85)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8', stepSize: 1 },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#f8fafc' },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderSeverityChart(breakdown) {
    const ctx = document.getElementById('chart-severity-breakdown');
    if (!ctx) return;

    if (severityChartInstance) {
        severityChartInstance.destroy();
    }

    severityChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Minor (< 40)', 'Major (40 - 74)', 'Critical (≥ 75)'],
            datasets: [{
                data: [breakdown.Minor || 0, breakdown.Major || 0, breakdown.Critical || 0],
                backgroundColor: ['#f59e0b', '#f97316', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#f8fafc', padding: 12 }
                }
            }
        }
    });
}

function renderRecentAccidentsTable(accidents) {
    const tbody = document.getElementById('tbl-recent-accidents');
    if (!tbody) return;

    if (!accidents || accidents.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">No vehicle crash records found. Upload a video to begin analysis.</td></tr>`;
        return;
    }

    tbody.innerHTML = accidents.map(acc => {
        const badgeClass = acc.severity_level === 'Critical' ? 'badge-critical' : (acc.severity_level === 'Major' ? 'badge-major' : 'badge-minor');
        const statusBadgeClass = acc.verification_status === 'confirmed' ? 'badge-confirmed' : 'badge-unverified';
        
        let hazardsText = [];
        if (acc.fire_detected) hazardsText.push('🔥 Fire');
        if (acc.smoke_detected) hazardsText.push('💨 Smoke');
        if (hazardsText.length === 0) hazardsText.push('None');

        const telemetryText = `Cars: ${acc.cars_count || 0} | Buses: ${acc.buses_count || 0} | Trucks: ${acc.trucks_count || 0} | Motorcycles: ${acc.bikes_count || 0}`;

        return `
            <tr>
                <td>#${acc.id}</td>
                <td>${acc.timestamp_sec}s</td>
                <td><span class="badge ${badgeClass}">${acc.severity_level}</span></td>
                <td style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">${acc.severity_score.toFixed(1)}</td>
                <td style="font-size: 0.8rem; color: var(--text-secondary);">${telemetryText}</td>
                <td>${hazardsText.join(', ')}</td>
                <td><span class="badge ${statusBadgeClass}">${acc.verification_status.toUpperCase()}</span></td>
                <td>
                    <a href="/accident/${acc.id}" class="btn btn-secondary btn-sm"><i class="fa-solid fa-eye"></i> Detail</a>
                </td>
            </tr>
        `;
    }).join('');
}

function loadAccidentHistory() {
    const severity = document.getElementById('filter-severity') ? document.getElementById('filter-severity').value : 'all';
    const status = document.getElementById('filter-status') ? document.getElementById('filter-status').value : 'all';

    fetch(`/api/v1/accidents?severity=${severity}&status=${status}`)
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('tbl-accident-history');
            if (!tbody) return;

            if (!data.accidents || data.accidents.length === 0) {
                tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 2rem;">No matching crash logs found.</td></tr>`;
                return;
            }

            tbody.innerHTML = data.accidents.map(acc => {
                const badgeClass = acc.severity_level === 'Critical' ? 'badge-critical' : (acc.severity_level === 'Major' ? 'badge-major' : 'badge-minor');
                const statusBadgeClass = acc.verification_status === 'confirmed' ? 'badge-confirmed' : 'badge-unverified';

                let hazardsText = [];
                if (acc.fire_detected) hazardsText.push('🔥 Fire');
                if (acc.smoke_detected) hazardsText.push('💨 Smoke');
                if (hazardsText.length === 0) hazardsText.push('None');

                const telemetryText = `Cars: ${acc.cars_count || 0} | Buses: ${acc.buses_count || 0} | Trucks: ${acc.trucks_count || 0} | Motorcycles: ${acc.bikes_count || 0}`;

                return `
                    <tr>
                        <td>#${acc.id}</td>
                        <td>Video #${acc.video_id}</td>
                        <td>${acc.timestamp_sec}s</td>
                        <td><span class="badge ${badgeClass}">${acc.severity_level}</span></td>
                        <td style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">${acc.severity_score.toFixed(1)}</td>
                        <td style="font-size: 0.8rem;">${telemetryText}</td>
                        <td>${hazardsText.join(', ')}</td>
                        <td><span class="badge ${statusBadgeClass}">${acc.verification_status.toUpperCase()}</span></td>
                        <td>
                            <a href="/accident/${acc.id}" class="btn btn-secondary btn-sm"><i class="fa-solid fa-arrow-right"></i> Inspect</a>
                        </td>
                    </tr>
                `;
            }).join('');
        });
}

function loadAccidentDetails(accidentId) {
    fetch(`/api/v1/accidents/${accidentId}`)
        .then(response => response.json())
        .then(acc => {
            document.getElementById('txt-acc-id').innerText = acc.id;
            document.getElementById('txt-severity-score').innerText = acc.severity_score.toFixed(1);
            document.getElementById('txt-acc-timestamp').innerText = `${acc.timestamp_sec}s`;
            document.getElementById('txt-acc-frame').innerText = acc.frame_number;
            document.getElementById('txt-acc-fire').innerText = acc.fire_detected ? 'Yes 🔥' : 'No';
            document.getElementById('txt-acc-smoke').innerText = acc.smoke_detected ? 'Yes 💨' : 'No';

            if (document.getElementById('txt-acc-cars')) document.getElementById('txt-acc-cars').innerText = acc.cars_count || 0;
            if (document.getElementById('txt-acc-buses')) document.getElementById('txt-acc-buses').innerText = acc.buses_count || 0;
            if (document.getElementById('txt-acc-trucks')) document.getElementById('txt-acc-trucks').innerText = acc.trucks_count || 0;
            if (document.getElementById('txt-acc-bikes')) document.getElementById('txt-acc-bikes').innerText = acc.bikes_count || 0;

            const badgeElem = document.getElementById('badge-severity-large');
            badgeElem.innerText = acc.severity_level;
            badgeElem.className = `badge ${acc.severity_level === 'Critical' ? 'badge-critical' : (acc.severity_level === 'Major' ? 'badge-major' : 'badge-minor')}`;

            const statusElem = document.getElementById('badge-verification-status');
            statusElem.innerText = acc.verification_status.toUpperCase();
            statusElem.className = `badge ${acc.verification_status === 'confirmed' ? 'badge-confirmed' : 'badge-unverified'}`;

            if (acc.snapshot_path) {
                document.getElementById('img-accident-snapshot').src = acc.snapshot_path.startsWith('/') ? acc.snapshot_path : '/' + acc.snapshot_path;
            }

            const tbody = document.getElementById('tbl-accident-vehicles');
            if (acc.vehicles && acc.vehicles.length > 0) {
                tbody.innerHTML = acc.vehicles.map(v => `
                    <tr>
                        <td>Track #${v.track_id || 'N/A'}</td>
                        <td>${v.vehicle_type.toUpperCase()}</td>
                        <td>${v.pre_impact_speed} km/h</td>
                        <td>${v.post_impact_speed} km/h</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">${v.impact_force_index}</td>
                    </tr>
                `).join('');
            } else {
                tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No involved track telemetry available.</td></tr>`;
            }
        });
}

function updateVerificationStatus(newStatus) {
    if (typeof ACCIDENT_ID === 'undefined') return;

    fetch(`/api/v1/accidents/${ACCIDENT_ID}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
    })
    .then(res => res.json())
    .then(acc => {
        loadAccidentDetails(acc.id);
    });
}
