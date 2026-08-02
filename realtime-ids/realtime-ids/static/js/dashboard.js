const token = localStorage.getItem('ids_token');
if (!token) {
    window.location.href = '/';
}

const MAX_POINTS = 20;
const cpuData = { labels: [], values: [] };
const memData = { labels: [], values: [] };

function makeChart(ctx, label, color) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label,
                data: [],
                borderColor: color,
                backgroundColor: color + '33',
                tension: 0.3,
                fill: true,
                pointRadius: 0,
            }]
        },
        options: {
            animation: false,
            scales: {
                y: { min: 0, max: 100, ticks: { color: '#94a3b8' } },
                x: { display: false }
            },
            plugins: { legend: { display: false } }
        }
    });
}

const cpuChart = makeChart(document.getElementById('cpu-chart'), 'CPU %', '#22c55e');
const memChart = makeChart(document.getElementById('memory-chart'), 'Memory %', '#38bdf8');

function pushPoint(chart, store, value) {
    const time = new Date().toLocaleTimeString();
    store.labels.push(time);
    store.values.push(value);
    if (store.labels.length > MAX_POINTS) {
        store.labels.shift();
        store.values.shift();
    }
    chart.data.labels = store.labels;
    chart.data.datasets[0].data = store.values;
    chart.update();
}

async function fetchMetrics() {
    try {
        const res = await fetch('/metrics', {
            headers: { Authorization: `Bearer ${token}` }
        });

        if (res.status === 401) {
            localStorage.removeItem('ids_token');
            window.location.href = '/';
            return;
        }

        const data = await res.json();
        const latest = data.latest_metrics || {};

        document.getElementById('cpu-value').textContent = `${latest.cpu ?? '--'}%`;
        document.getElementById('memory-value').textContent = `${latest.memory ?? '--'}%`;

        pushPoint(cpuChart, cpuData, latest.cpu ?? 0);
        pushPoint(memChart, memData, latest.memory ?? 0);

        const banner = document.getElementById('anomaly-banner');
        if (latest.anomaly) {
            banner.textContent = `Anomaly detected: ${latest.anomaly}`;
            banner.className = 'banner alert';
        } else {
            banner.textContent = 'No active anomalies';
            banner.className = 'banner ok';
        }

        const list = document.getElementById('anomaly-list');
        list.innerHTML = '';
        (data.recent_anomalies || []).slice().reverse().forEach((a) => {
            const li = document.createElement('li');
            li.textContent = `[${a.logged_at}] ${a.message}`;
            list.appendChild(li);
        });
    } catch (err) {
        console.error('Failed to fetch metrics', err);
    }
}

fetchMetrics();
setInterval(fetchMetrics, 5000);
