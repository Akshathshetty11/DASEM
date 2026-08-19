// Video Upload & Polling Script

function handleFileSelected(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('upload-status-text');
    const percentText = document.getElementById('upload-percent');

    progressContainer.style.display = 'block';
    statusText.innerText = 'Uploading video file...';
    progressBar.style.width = '20%';
    percentText.innerText = '20%';

    fetch('/api/v1/videos/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert('Upload failed: ' + data.error);
            progressContainer.style.display = 'none';
            return;
        }

        statusText.innerText = 'Analyzing Cars, Buses, Trucks, Bikes, Humans, Fire & Smoke...';
        progressBar.style.width = '60%';
        percentText.innerText = '60%';

        pollVideoStatus(data.video.id);
    })
    .catch(err => {
        alert('Upload failed: ' + err);
        progressContainer.style.display = 'none';
    });
}

function pollVideoStatus(videoId) {
    const interval = setInterval(() => {
        fetch(`/api/v1/videos/${videoId}/status`)
            .then(res => res.json())
            .then(data => {
                const status = data.status;
                const statusBadge = document.getElementById('badge-processing-status');
                
                if (statusBadge) {
                    statusBadge.innerText = status.toUpperCase();
                    statusBadge.className = `badge ${status === 'completed' ? 'badge-confirmed' : 'badge-unverified'}`;
                }

                if (data.video) {
                    document.getElementById('txt-video-filename').innerText = data.video.filename;
                    document.getElementById('txt-video-duration').innerText = `${data.video.duration_seconds}s`;
                    document.getElementById('txt-video-fps').innerText = `${data.video.fps} FPS`;
                    document.getElementById('txt-video-accidents').innerText = `${data.accident_count} Incident(s)`;

                    const summary = data.detection_summary || {};
                    const summaryTxt = `Cars: ${summary.cars || 0} | Buses: ${summary.buses || 0} | Trucks: ${summary.trucks || 0} | Bikes: ${summary.bikes || 0} | Humans: ${summary.persons || 0}`;
                    if (document.getElementById('txt-video-objects')) {
                        document.getElementById('txt-video-objects').innerText = summaryTxt;
                    }
                }

                if (status === 'completed') {
                    clearInterval(interval);
                    document.getElementById('upload-status-text').innerText = 'Multi-object analysis complete!';
                    document.getElementById('upload-progress-bar').style.width = '100%';
                    document.getElementById('upload-percent').innerText = '100%';
                    loadUploadedVideos();
                } else if (status === 'failed') {
                    clearInterval(interval);
                    document.getElementById('upload-status-text').innerText = 'Processing failed.';
                }
            })
            .catch(err => console.error('Status poll error:', err));
    }, 2000);
}

function loadUploadedVideos() {
    fetch('/api/v1/videos')
        .then(res => res.json())
        .then(videos => {
            const tbody = document.getElementById('tbl-uploaded-videos');
            if (!tbody) return;

            if (!videos || videos.length === 0) {
                tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">No videos uploaded yet.</td></tr>`;
                return;
            }

            tbody.innerHTML = videos.map(v => {
                const statusBadgeClass = v.status === 'completed' ? 'badge-confirmed' : 'badge-unverified';
                const uploadedDate = v.uploaded_at ? new Date(v.uploaded_at).toLocaleString() : '--';
                const videoUrl = v.processed_path ? (v.processed_path.startsWith('/') ? v.processed_path : '/' + v.processed_path) : '#';
                const summary = v.object_counts || {};

                return `
                    <tr>
                        <td>#${v.id}</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">${v.filename}</td>
                        <td>${uploadedDate}</td>
                        <td>${v.duration_seconds ? v.duration_seconds + 's' : '--'}</td>
                        <td><span class="badge ${statusBadgeClass}">${v.status.toUpperCase()}</span></td>
                        <td style="font-size: 0.8rem; color: var(--text-secondary);">
                            Cars: ${summary.cars || 0} | Buses: ${summary.buses || 0} | Trucks: ${summary.trucks || 0} | Bikes: ${summary.bikes || 0} | Humans: ${summary.persons || 0}
                        </td>
                        <td style="font-weight: 600; color: ${v.accident_count > 0 ? 'var(--severity-major)' : 'var(--text-muted)'};">${v.accident_count}</td>
                        <td>
                            ${v.processed_path ? `<a href="${videoUrl}" target="_blank" class="btn btn-secondary btn-sm"><i class="fa-solid fa-play"></i> Watch Output</a>` : '--'}
                        </td>
                    </tr>
                `;
            }).join('');
        });
}
