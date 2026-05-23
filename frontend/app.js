const API_BASE = 'http://localhost:8000';
const PAGE_SIZE = 6;

// Dữ liệu phân trang
let _logsData = [];
let _logsPage = 1;
let _violationsData = [];
let _violationsPage = 1;

// Khởi tạo khi trang tải xong
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    initClock();
    initNavigation();
    initScrollAnimations();
    initCamera();
    initSearch();
    initImageModal();
    loadStats();
    loadLogs();
    loadViolations();
    loadRecentDetections();

    // Auto refresh mỗi 10 giây
    setInterval(() => {
        loadStats();
        loadRecentDetections();
    }, 10000);
});

// Đồng hồ realtime
function initClock() {
    const el = document.getElementById('clock-time');
    function update() {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('vi-VN', { hour12: false });
    }
    update();
    setInterval(update, 1000);
}

// Điều hướng navbar
function initNavigation() {
    const toggle = document.getElementById('nav-toggle');
    const links = document.getElementById('nav-links');
    if (toggle) toggle.addEventListener('click', () => links.classList.toggle('open'));

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            links.classList.remove('open');
            const target = document.querySelector(link.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    // Highlight nav link khi scroll
    const sections = document.querySelectorAll('section[id]');
    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY + 100;
        sections.forEach(s => {
            const top = s.offsetTop, h = s.offsetHeight, id = s.id;
            if (scrollY >= top && scrollY < top + h) {
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                const active = document.querySelector(`.nav-link[data-section="${id}"]`);
                if (active) active.classList.add('active');
            }
        });
        document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 50);
    });
}

// Animation khi scroll vào viewport
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
}

// Hiệu ứng đếm số
function animateCounter(el, target) {
    const duration = 1500;
    const start = parseInt(el.textContent) || 0;
    const diff = target - start;
    if (diff === 0) return;
    const startTime = performance.now();
    function step(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(start + diff * ease).toLocaleString('vi-VN');
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// Tải thống kê tổng quan
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/api/stats`);
        const json = await res.json();
        if (json.status === 'success') {
            const d = json.data;
            animateCounter(document.getElementById('stat-total-num'), d.total_vehicles);
            animateCounter(document.getElementById('stat-violations-num'), d.total_violations_detected);
            animateCounter(document.getElementById('stat-plates-num'), d.unique_plates);
            animateCounter(document.getElementById('stat-today-num'), d.today_count);
            updateStatus(true);
        }
    } catch (e) {
        console.warn('Stats API unavailable:', e.message);
        updateStatus(false);
    }
}

// Tạo thanh phân trang
function renderPagination(containerId, currentPage, totalPages, onPageChange) {
    const container = document.getElementById(containerId);
    if (!container || totalPages <= 1) { if (container) container.innerHTML = ''; return; }

    let html = '';
    html += `<button class="page-btn ${currentPage === 1 ? 'disabled' : ''}" data-page="${currentPage - 1}"><i data-lucide="chevron-left"></i></button>`;

    // Hiện trang đầu, cuối, và xung quanh trang hiện tại
    const pages = [];
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= currentPage - 1 && i <= currentPage + 1)) {
            pages.push(i);
        } else if (pages[pages.length - 1] !== '...') {
            pages.push('...');
        }
    }
    pages.forEach(p => {
        if (p === '...') {
            html += `<span class="page-info">...</span>`;
        } else {
            html += `<button class="page-btn ${p === currentPage ? 'active' : ''}" data-page="${p}">${p}</button>`;
        }
    });

    html += `<button class="page-btn ${currentPage === totalPages ? 'disabled' : ''}" data-page="${currentPage + 1}"><i data-lucide="chevron-right"></i></button>`;

    container.innerHTML = html;
    lucide.createIcons();

    container.querySelectorAll('.page-btn:not(.disabled)').forEach(btn => {
        btn.addEventListener('click', () => {
            const page = parseInt(btn.dataset.page);
            if (page >= 1 && page <= totalPages) onPageChange(page);
        });
    });
}

// Render bảng lịch sử nhận diện (có phân trang)
function renderLogsPage(page) {
    _logsPage = page;
    const tbody = document.getElementById('logs-tbody');
    const totalPages = Math.ceil(_logsData.length / PAGE_SIZE);
    const start = (page - 1) * PAGE_SIZE;
    const pageData = _logsData.slice(start, start + PAGE_SIZE);

    tbody.innerHTML = '';
    pageData.forEach((row, i) => {
        const tr = document.createElement('tr');
        tr.style.animationDelay = `${i * 0.03}s`;
        tr.className = 'fade-in-row';
        const time = new Date(row.thoi_gian_chup).toLocaleString('vi-VN');
        const badge = row.co_vi_pham
            ? '<span class="badge-danger"><i data-lucide="alert-triangle"></i> Có</span>'
            : '<span class="badge-success"><i data-lucide="check-circle"></i> Không</span>';
        tr.innerHTML = `
            <td>${start + i + 1}</td>
            <td><span class="plate-tag">${row.bien_so || '---'}</span></td>
            <td>${time}</td>
            <td>${row.link_anh_goc ? `<a href="#" onclick="openImageModal('${row.link_anh_goc}','Ảnh gốc — ${row.bien_so || ''}');return false" class="img-link"><i data-lucide="image"></i> Xem</a>` : '—'}</td>
            <td>${row.link_anh_bien_so ? `<a href="#" onclick="openImageModal('${row.link_anh_bien_so}','Biển số — ${row.bien_so || ''}');return false" class="img-link"><i data-lucide="image"></i> Xem</a>` : '—'}</td>
            <td>${badge}</td>`;
        tbody.appendChild(tr);
    });
    lucide.createIcons();
    renderPagination('logs-pagination', page, totalPages, renderLogsPage);
}

async function loadLogs() {
    const tbody = document.getElementById('logs-tbody');
    const loading = document.getElementById('logs-loading');
    const empty = document.getElementById('logs-empty');
    const limit = document.getElementById('log-limit').value;

    loading.style.display = 'flex';
    empty.style.display = 'none';
    tbody.innerHTML = '';
    document.getElementById('logs-pagination').innerHTML = '';

    try {
        const res = await fetch(`${API_BASE}/api/logs?limit=${limit}`);
        const json = await res.json();
        loading.style.display = 'none';

        if (json.status === 'success' && json.data.length > 0) {
            _logsData = json.data;
            _logsPage = 1;
            document.getElementById('table-count').textContent = `${json.data.length} bản ghi`;
            renderLogsPage(1);
        } else {
            _logsData = [];
            empty.style.display = 'flex';
        }
    } catch (e) {
        loading.style.display = 'none';
        empty.style.display = 'flex';
        empty.querySelector('p').textContent = 'Không thể kết nối API';
    }
}

document.getElementById('btn-refresh-logs')?.addEventListener('click', loadLogs);
document.getElementById('log-limit')?.addEventListener('change', loadLogs);

// Render bảng phạt nguội (có phân trang)
function renderViolationsPage(page) {
    _violationsPage = page;
    const tbody = document.getElementById('violations-tbody');
    const totalPages = Math.ceil(_violationsData.length / PAGE_SIZE);
    const start = (page - 1) * PAGE_SIZE;
    const pageData = _violationsData.slice(start, start + PAGE_SIZE);

    tbody.innerHTML = '';
    pageData.forEach((row, i) => {
        const tr = document.createElement('tr');
        tr.style.animationDelay = `${i * 0.05}s`;
        tr.className = 'fade-in-row';
        const time = new Date(row.thoi_gian_vi_pham).toLocaleString('vi-VN');
        const money = Number(row.so_tien).toLocaleString('vi-VN') + ' ₫';
        const statusText = (row.trang_thai || '').toLowerCase();
        const isPaid = statusText.includes('da nop') || statusText.includes('đã nộp');
        const statusCls = isPaid ? 'badge-success' : 'badge-danger';
        const statusLabel = isPaid ? 'Đã nộp phạt' : 'Chưa nộp phạt';
        tr.innerHTML = `
            <td>${start + i + 1}</td>
            <td><span class="plate-tag warning">${row.bien_so}</span></td>
            <td>${row.loi_vi_pham}</td>
            <td class="money">${money}</td>
            <td><span class="${statusCls}">${statusLabel}</span></td>
            <td>${time}</td>`;
        tbody.appendChild(tr);
    });
    lucide.createIcons();
    renderPagination('violations-pagination', page, totalPages, renderViolationsPage);
}

async function loadViolations() {
    const tbody = document.getElementById('violations-tbody');
    const loading = document.getElementById('violations-loading');
    const empty = document.getElementById('violations-empty');

    loading.style.display = 'flex';
    empty.style.display = 'none';
    tbody.innerHTML = '';
    document.getElementById('violations-pagination').innerHTML = '';

    try {
        const res = await fetch(`${API_BASE}/api/violations`);
        const json = await res.json();
        loading.style.display = 'none';

        if (json.status === 'success' && json.data.length > 0) {
            _violationsData = json.data;
            _violationsPage = 1;
            document.getElementById('violations-count').textContent = `${json.data.length} vi phạm`;
            renderViolationsPage(1);
        } else {
            _violationsData = [];
            empty.style.display = 'flex';
        }
    } catch (e) {
        loading.style.display = 'none';
        empty.style.display = 'flex';
    }
}

document.getElementById('btn-refresh-violations')?.addEventListener('click', loadViolations);

// Sidebar nhận diện gần đây
async function loadRecentDetections() {
    const list = document.getElementById('recent-list');
    try {
        const res = await fetch(`${API_BASE}/api/logs?limit=6`);
        const json = await res.json();
        if (json.status === 'success' && json.data.length > 0) {
            list.innerHTML = json.data.map(row => {
                const time = new Date(row.thoi_gian_chup).toLocaleTimeString('vi-VN', { hour12: false });
                const dot = row.co_vi_pham ? 'red' : 'green';
                return `<div class="recent-item">
                    <div class="recent-dot ${dot}"></div>
                    <div class="recent-info">
                        <span class="recent-plate">${row.bien_so || '---'}</span>
                        <span class="recent-time">${time}</span>
                    </div>
                    <span class="recent-status ${dot}">${row.co_vi_pham ? 'VP' : 'OK'}</span>
                </div>`;
            }).join('');
        } else {
            list.innerHTML = '<div class="recent-empty">Chưa có dữ liệu</div>';
        }
    } catch {
        list.innerHTML = '<div class="recent-empty">Chờ kết nối API...</div>';
    }
}

document.getElementById('btn-refresh-recent')?.addEventListener('click', loadRecentDetections);

// Tìm kiếm biển số
function initSearch() {
    const input = document.getElementById('search-input');
    const btn = document.getElementById('btn-search');
    btn?.addEventListener('click', () => doSearch(input.value));
    input?.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(input.value); });

    document.querySelectorAll('.hint-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            input.value = tag.dataset.plate;
            doSearch(tag.dataset.plate);
        });
    });
}

async function doSearch(query) {
    if (!query.trim()) return showToast('Vui lòng nhập biển số xe', 'warning');
    const card = document.getElementById('search-results-card');
    const tbody = document.getElementById('search-tbody');
    const empty = document.getElementById('search-empty');
    document.getElementById('search-query-display').textContent = query;
    card.style.display = 'block';
    tbody.innerHTML = '';
    empty.style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/api/search?plate=${encodeURIComponent(query)}`);
        const json = await res.json();
        if (json.status === 'success' && json.data.length > 0) {
            document.getElementById('search-result-count').textContent = `${json.data.length} kết quả`;
            json.data.forEach((row, i) => {
                const tr = document.createElement('tr');
                tr.className = 'fade-in-row';
                tr.style.animationDelay = `${i * 0.04}s`;
                const time = new Date(row.thoi_gian_chup).toLocaleString('vi-VN');
                const badge = row.co_vi_pham
                    ? '<span class="badge-danger">Có</span>'
                    : '<span class="badge-success">Không</span>';
                tr.innerHTML = `
                    <td>${i + 1}</td>
                    <td><span class="plate-tag">${row.bien_so}</span></td>
                    <td>${time}</td>
                    <td>${row.link_anh_goc ? `<a href="#" onclick="openImageModal('${row.link_anh_goc}','Ảnh gốc — ${row.bien_so}');return false" class="img-link">Xem</a>` : '—'}</td>
                    <td>${row.link_anh_bien_so ? `<a href="#" onclick="openImageModal('${row.link_anh_bien_so}','Biển số — ${row.bien_so}');return false" class="img-link">Xem</a>` : '—'}</td>
                    <td>${badge}</td>`;
                tbody.appendChild(tr);
            });
            lucide.createIcons();
        } else {
            empty.style.display = 'flex';
            document.getElementById('search-result-count').textContent = '0 kết quả';
        }
    } catch {
        empty.style.display = 'flex';
        empty.querySelector('p').textContent = 'Lỗi kết nối API';
    }
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Camera (Webcam + Video file)
function initCamera() {
    const video = document.getElementById('camera-video');
    const placeholder = document.getElementById('camera-placeholder');
    const btnStart = document.getElementById('btn-start-camera');
    const btnStop = document.getElementById('btn-stop-camera');
    const btnLoadVideo = document.getElementById('btn-load-video');
    const fileInput = document.getElementById('video-file-input');

    function showVideo() {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }
        placeholder.style.display = 'none';
        video.style.display = 'block';
        btnStop.disabled = false;
        document.getElementById('cam-status-text').innerHTML = '<i data-lucide="wifi"></i> Đang phát';
        lucide.createIcons();
    }

    // Mở Webcam
    btnStart?.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
            video.srcObject = stream;
            video.removeAttribute('src');
            video.loop = false;
            video.controls = false;
            video.muted = true;
            video.playsInline = true;
            await video.play();
            showVideo();
            btnStart.disabled = true;
            showToast('Camera đã được kích hoạt', 'success');

            video.onloadedmetadata = () => {
                const res = `${video.videoWidth}×${video.videoHeight}`;
                document.querySelector('.cam-res').innerHTML = `<i data-lucide="monitor"></i> ${res}`;
                lucide.createIcons();
            };
        } catch (err) {
            console.error('Camera error:', err);
            showToast('Không thể truy cập Camera: ' + err.message, 'error');
        }
    });

    // Tải video từ máy
    btnLoadVideo?.addEventListener('click', () => fileInput.click());

    fileInput?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const url = URL.createObjectURL(file);
        video.srcObject = null;
        video.src = url;
        video.loop = true;
        video.controls = true;
        video.muted = true;
        video.play();
        showVideo();
        btnStart.disabled = false;
        showToast(`Đã tải video: ${file.name}`, 'success');

        video.onloadeddata = () => {
            const res = `${video.videoWidth}×${video.videoHeight}`;
            document.querySelector('.cam-res').innerHTML = `<i data-lucide="monitor"></i> ${res}`;
            lucide.createIcons();
        };
    });

    // Dừng camera/video
    btnStop?.addEventListener('click', () => {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }
        video.pause();
        video.removeAttribute('src');
        video.load();
        video.controls = false;

        video.style.display = 'none';
        placeholder.style.display = 'flex';
        btnStart.disabled = false;
        btnStop.disabled = true;
        fileInput.value = '';
        document.getElementById('cam-status-text').innerHTML = '<i data-lucide="wifi-off"></i> Đã tắt';
        lucide.createIcons();
    });

    // Fullscreen
    document.getElementById('btn-fullscreen')?.addEventListener('click', () => {
        const vp = document.getElementById('camera-viewport');
        if (vp.requestFullscreen) vp.requestFullscreen();
    });
}

// Trạng thái hệ thống
function updateStatus(online) {
    const el = document.getElementById('system-status');
    if (!el) return;
    el.querySelector('.status-dot').className = `status-dot ${online ? 'online' : 'offline'}`;
    el.querySelector('.status-text').textContent = online ? 'Hệ thống hoạt động' : 'Mất kết nối';
}

// Toast thông báo
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { success: 'check-circle', error: 'x-circle', warning: 'alert-triangle', info: 'info' };
    toast.innerHTML = `<i data-lucide="${icons[type] || 'info'}"></i><span>${message}</span>`;
    container.appendChild(toast);
    lucide.createIcons();
    setTimeout(() => { toast.classList.add('hide'); setTimeout(() => toast.remove(), 400); }, 3500);
}

// Modal xem ảnh
function initImageModal() {
    const modal = document.getElementById('image-modal');
    const closeBtn = document.getElementById('modal-close');

    closeBtn?.addEventListener('click', () => modal.classList.remove('active'));
    modal?.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') modal?.classList.remove('active');
    });
}

function openImageModal(url, caption) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-image');
    const cap = document.getElementById('modal-caption');
    img.src = url;
    cap.textContent = caption || '';
    modal.classList.add('active');
}
