/* ================= UX UTILITIES ================= */
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const _animFrames = new WeakMap();

/** Smoothly counts a number element from its current value to `to`. */
function animateValue(el, to, { decimals = 0, duration = 550, suffix = '' } = {}) {
    if (!el) return;
    const from = parseFloat((el.textContent || '0').replace(/[^0-9.\-]/g, '')) || 0;
    if (prefersReducedMotion || Math.abs(to - from) < (decimals ? 0.05 : 1)) {
        el.textContent = to.toFixed(decimals) + suffix;
        return;
    }
    if (_animFrames.has(el)) cancelAnimationFrame(_animFrames.get(el));
    const start = performance.now();
    const ease = t => 1 - Math.pow(1 - t, 3);
    function tick(now) {
        const p = Math.min(1, (now - start) / duration);
        const val = from + (to - from) * ease(p);
        el.textContent = (decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + suffix;
        if (p < 1) {
            _animFrames.set(el, requestAnimationFrame(tick));
        } else {
            el.textContent = (decimals ? to.toFixed(decimals) : to.toLocaleString()) + suffix;
        }
    }
    _animFrames.set(el, requestAnimationFrame(tick));
}

/** Briefly flashes an element to draw attention to value updates. */
function flashValue(el) {
    if (!el || prefersReducedMotion) return;
    el.classList.remove('value-flash');
    void el.offsetWidth;
    el.classList.add('value-flash');
}

/** Pushes a live alert into the toast feed */
function showToast(title, body, color = '#a855f7') {
    const stack = document.getElementById('toastStack');
    if (!stack) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <span class="toast-dot" style="background:${color}; box-shadow:0 0 8px ${color};"></span>
        <div>
            <div class="toast-title" style="color:${color};">${title}</div>
            <div class="toast-body">${body}</div>
        </div>`;
    stack.appendChild(toast);
    while (stack.children.length > 4) stack.removeChild(stack.firstChild);
    setTimeout(() => {
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
    }, 5000);
}

/* ================= ATTACK FORECAST TREE ENGINE ================= */
const TREE_DATA = [
    { id:0,  parent:null, name:"Suspicious Network Activity", prob:100, conf:98, desc:"Anomalous behaviour deviating from baseline network traffic patterns." },
    { id:1,  parent:0, name:"Initial Access",       prob:64, conf:82, desc:"Adversary attempts to gain an initial foothold into the network." },
    { id:2,  parent:0, name:"Discovery",             prob:23, desc:"Attacker enumerates internal hosts, services and network topology." },
    { id:3,  parent:0, name:"Credential Access",     prob:13, desc:"Attempt to steal account credentials for further access." },
    { id:4,  parent:1, name:"Execution",             prob:71, desc:"Malicious code is executed on the compromised host." },
    { id:5,  parent:1, name:"Persistence",           prob:29, desc:"Attacker maintains footholds across system restarts or credential changes." },
    { id:6,  parent:2, name:"Lateral Movement",      prob:55, desc:"Pivoting to other hosts using discovered network information." },
    { id:7,  parent:2, name:"Collection",            prob:45, desc:"Gathering data of interest from discovered internal sources." },
    { id:8,  parent:3, name:"Privilege Escalation",  prob:67, desc:"Using stolen credentials to gain higher-level permissions." },
    { id:9,  parent:3, name:"Lateral Movement",      prob:33, desc:"Moving across the network using compromised credentials." },
    { id:10, parent:4, name:"Privilege Escalation",  prob:58, desc:"Exploiting executed code context to gain elevated privileges." },
    { id:11, parent:4, name:"Defense Evasion",       prob:42, desc:"Techniques used to avoid detection during execution." },
    { id:12, parent:5, name:"Defense Evasion",       prob:50, desc:"Hiding persistence mechanisms from security tooling." },
    { id:13, parent:5, name:"Credential Access",     prob:50, desc:"Harvesting credentials from the persistent foothold." },
    { id:14, parent:6, name:"Command and Control",   prob:62, desc:"Establishing outbound channel for remote operator control." },
    { id:15, parent:6, name:"Exfiltration",          prob:38, desc:"Extracting collected data from the compromised host directly." },
    { id:16, parent:7, name:"Exfiltration",          prob:80, desc:"Bulk transfer of collected sensitive data out of the network." },
    { id:17, parent:8, name:"Lateral Movement",      prob:70, desc:"Elevated access used to pivot deeper into the network." },
    { id:18, parent:8, name:"Defense Evasion",       prob:30, desc:"Elevated privileges used to disable security controls." },
    { id:19, parent:9, name:"Command and Control",   prob:55, desc:"Compromised host establishes covert C2 communication." },
    { id:20, parent:9, name:"Exfiltration",          prob:45, desc:"Sensitive data extracted using existing lateral access." },
    { id:21, parent:10, name:"Credential Access",    prob:64, desc:"Privileged access leveraged to dump further credentials." },
    { id:22, parent:10, name:"Lateral Movement",     prob:36, desc:"Escalated privileges enable movement to critical systems." },
    { id:23, parent:14, name:"Exfiltration",         prob:75, desc:"Data is exfiltrated through the established C2 channel." },
    { id:24, parent:14, name:"Impact",               prob:25, desc:"Operator uses C2 access to disrupt or destroy systems/data." },
    { id:25, parent:17, name:"Command and Control",  prob:58, desc:"Newly reached host establishes a secondary C2 channel." },
    { id:26, parent:17, name:"Collection",           prob:42, desc:"Sensitive data gathered from newly accessed internal systems." },
];

const BOX_W = 200, BOX_H = 62, COL_GAP = 92, ROW_HEIGHT = 76, PAD = 40;

function buildTreeAndRender() {
    const nodeMap = {};
    TREE_DATA.forEach(n => nodeMap[n.id] = { ...n, children: [] });
    const childrenOf = {};
    TREE_DATA.forEach(n => {
        if (n.parent !== null) {
            childrenOf[n.parent] = childrenOf[n.parent] || [];
            childrenOf[n.parent].push(nodeMap[n.id]);
            nodeMap[n.parent].children.push(nodeMap[n.id]);
        }
    });
    const root = nodeMap[0];

    (function assignDepth(node, depth) {
        node.depth = depth;
        node.children.forEach(c => assignDepth(c, depth + 1));
    })(root, 0);

    let leafCounter = 0;
    (function assignRow(node) {
        if (node.children.length === 0) {
            node.row = leafCounter;
            leafCounter += 1;
        } else {
            node.children.forEach(assignRow);
            const rows = node.children.map(c => c.row);
            node.row = (Math.min(...rows) + Math.max(...rows)) / 2;
        }
    })(root);

    Object.values(childrenOf).forEach(childArr => {
        let top = childArr[0];
        childArr.forEach(c => { if (c.prob > top.prob) top = c; });
        top.isTop = true;
    });

    const flat = Object.values(nodeMap);
    let maxDepth = 0, maxRow = 0;
    flat.forEach(n => {
        n.x = PAD + n.depth * (BOX_W + COL_GAP);
        n.y = PAD + n.row * ROW_HEIGHT;
        if (n.depth > maxDepth) maxDepth = n.depth;
        if (n.row > maxRow) maxRow = n.row;
    });

    const contentW = PAD * 2 + BOX_W + maxDepth * (BOX_W + COL_GAP);
    const contentH = PAD * 2 + BOX_H + maxRow * ROW_HEIGHT;

    const canvas = document.getElementById('forecastTreeCanvas');
    const svg = document.getElementById('forecastTreeSvg');
    if (!canvas || !svg) return;
    
    canvas.style.width = contentW + 'px';
    canvas.style.height = contentH + 'px';
    svg.setAttribute('width', contentW);
    svg.setAttribute('height', contentH);

    flat.forEach(n => {
        if (n.parent === null) return;
        const p = nodeMap[n.parent];
        const x1 = p.x + BOX_W, y1 = p.y + BOX_H / 2;
        const x2 = n.x, y2 = n.y + BOX_H / 2;
        const midX = x1 + (x2 - x1) / 2;
        const d = `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d);
        path.setAttribute('fill', 'none');
        path.setAttribute('class', 'tree-line');
        path.dataset.child = n.id;

        if (n.isTop) {
            path.setAttribute('stroke', 'rgba(129, 140, 248,0.55)');
            path.setAttribute('stroke-width', '2.4');
        } else {
            path.setAttribute('stroke', 'rgba(168,85,247,0.28)');
            path.setAttribute('stroke-width', '1.4');
        }
        svg.appendChild(path);
    });

    flat.forEach(n => {
        const box = document.createElement('div');
        box.className = 'tree-box';
        box.style.left = n.x + 'px';
        box.style.top = n.y + 'px';
        box.style.width = BOX_W + 'px';
        box.style.height = BOX_H + 'px';
        box.dataset.id = n.id;

        if (n.parent === null) box.classList.add('tree-box-root');
        else if (n.isTop) box.classList.add('tree-box-top');

        let inner = '';
        if (n.parent === null) inner += `<div class="root-pill">Current State</div>`;
        inner += `<div class="stage-name">${n.name}</div>`;
        inner += `<div class="stage-prob">Probability: ${n.prob}%</div>`;
        if (n.conf) inner += `<div class="stage-conf">Confidence: ${n.conf}%</div>`;
        box.innerHTML = inner;

        box.addEventListener('mouseenter', () => highlightPath(n.id, nodeMap));
        box.addEventListener('mouseleave', clearHighlight);

        canvas.appendChild(box);
    });
}

function highlightPath(id, nodeMap) {
    const ancestors = new Set();
    let cur = nodeMap[id];
    while (cur) {
        ancestors.add(cur.id);
        cur = cur.parent !== null ? nodeMap[cur.parent] : null;
    }

    document.querySelectorAll('.tree-box').forEach(box => {
        const bid = parseInt(box.dataset.id);
        box.classList.toggle('highlight', ancestors.has(bid));
        box.classList.toggle('dim', !ancestors.has(bid));
    });

    document.querySelectorAll('.tree-line').forEach(line => {
        const cid = parseInt(line.dataset.child);
        line.classList.toggle('highlight', ancestors.has(cid));
        line.classList.toggle('dim', !ancestors.has(cid));
    });

    const node = nodeMap[id];
    document.getElementById('treeInfoText').innerHTML =
        `<span style="color:#c084fc; font-weight:700;">${node.name}</span> — Probability: <span style="color:#818cf8; font-weight:700;">${node.prob}%</span>${node.conf ? ` · Confidence: <span style="color:#818cf8;">${node.conf}%</span>` : ''} <br><span style="color:#948ba8;">${node.desc}</span>`;
}

function clearHighlight() {
    document.querySelectorAll('.tree-box').forEach(b => { b.classList.remove('highlight', 'dim'); });
    document.querySelectorAll('.tree-line').forEach(l => { l.classList.remove('highlight', 'dim'); });
    document.getElementById('treeInfoText').textContent = 'Hover over any predicted state above to trace its attack path and view details.';
}

/* ================= LIVE DASHBOARD & WEBSOCKET ================= */
let STATE = {
    event: "Detected: Benign",
    risk: 0,
    next_attack: "Benign",
    mitre: "Reconnaissance",
    confidence: 0,
    top_triggers: [],
    flows: 0,
    packets: 0,
    anomaly: 0,
    status: "MONITORING",
    trafficHistory: []
};

for (let i = 0; i < 20; i++) {
    STATE.trafficHistory.push({ flows: 0, packets: 0 });
}

let CHARTS = {};
const rand = (min, max) => Math.random() * (max - min) + min;
const formatTime = () => new Date().toLocaleTimeString("en-US", { hour12: false });

function initCharts() {
    const ctx = document.getElementById('trafficReportChart');
    if (!ctx) return;
    CHARTS.traffic = new Chart(ctx, {
        type: 'line',
        data: {
            labels: STATE.trafficHistory.map((_, i) => i),
            datasets: [
                {
                    label: 'Flows/Sec',
                    data: STATE.trafficHistory.map(d => d.flows),
                    borderColor: '#818cf8',
                    backgroundColor: 'rgba(129, 140, 248, 0.10)',
                    tension: 0.4, fill: true, pointRadius: 0, borderWidth: 2.5
                },
                {
                    label: 'Packets/Sec',
                    data: STATE.trafficHistory.map(d => d.packets),
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.10)',
                    tension: 0.4, fill: true, pointRadius: 0, borderWidth: 2.5
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(13,8,21,0.9)',
                    borderColor: 'rgba(168,85,247,0.4)', borderWidth: 1,
                    titleFont: { family: 'Rajdhani' }, bodyFont: { family: 'Rajdhani' }
                }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(168,85,247,0.08)' }, ticks: { color: '#948ba8', font: { family: 'Rajdhani' } } },
                x: { grid: { display: false }, ticks: { color: '#948ba8', font: { family: 'Rajdhani' } } }
            }
        }
    });
}

function updateTrafficChart() {
    if (!CHARTS.traffic) return;
    CHARTS.traffic.data.labels = STATE.trafficHistory.map((_, i) => i);
    CHARTS.traffic.data.datasets[0].data = STATE.trafficHistory.map(d => d.flows);
    CHARTS.traffic.data.datasets[1].data = STATE.trafficHistory.map(d => d.packets);
    CHARTS.traffic.update('none');
}

function updateUI() {
    animateValue(document.getElementById("flowsPerSec"), STATE.flows, { duration: 400 });
    animateValue(document.getElementById("packetsPerSec"), STATE.packets, { duration: 400 });

    // Triggers / Events list
    const triggers = STATE.top_triggers || [];
    animateValue(document.getElementById("eventCount"), triggers.length, { duration: 300 });
    let eventHTML = "";
    if (triggers.length > 0) {
        triggers.forEach((trig, idx) => {
            const color = idx === 0 ? "#fb7185" : "#fbbf24";
            eventHTML += `<div class="slide-up text-xs flex justify-between gap-2 font-semibold" style="color:${color}">
                <span>${trig}</span>
                <span class="mono-num text-purple-300/30" style="font-size:10px;">${formatTime()}</span>
            </div>`;
        });
    } else {
        eventHTML = `<div class="text-xs text-purple-300/40">✓ ${STATE.event || "Monitoring normal traffic"}</div>`;
    }
    document.getElementById("eventsList").innerHTML = eventHTML;

    // Status Card & Threat Level
    let status = STATE.status || "MONITORING";
    let statusColor, glowClass, threatLabel;
    
    if (STATE.risk < 35) {
        statusColor = "#34d399"; glowClass = "glow-green"; threatLabel = "LOW";
    } else if (STATE.risk <= 70) {
        statusColor = "#fbbf24"; glowClass = "glow-amber"; threatLabel = "MEDIUM";
    } else {
        statusColor = "#fb7185"; glowClass = "glow-red"; threatLabel = "HIGH";
    }

    const indicator = document.getElementById("statusIndicator");
    if (indicator) {
        indicator.style.background = statusColor;
        indicator.style.boxShadow = `0 0 10px ${statusColor}`;
    }

    const statusLabelEl = document.getElementById("statusLabel");
    if (statusLabelEl) {
        statusLabelEl.style.color = statusColor;
        if (statusLabelEl.textContent !== status) {
            statusLabelEl.textContent = status;
            flashValue(statusLabelEl);
        }
    }

    const descEl = document.getElementById("statusDesc");
    if (descEl) descEl.textContent = STATE.event || "All systems operating normally";

    animateValue(document.getElementById("riskScoreVal"), Math.round(STATE.risk), { duration: 400 });

    const statusCard = document.getElementById("statusCardGlow");
    if (statusCard) {
        statusCard.classList.remove("glow-green", "glow-amber", "glow-red");
        statusCard.classList.add(glowClass);
    }

    const threatLabelEl = document.getElementById("threatLevelLabel");
    if (threatLabelEl) {
        if (threatLabelEl.textContent !== threatLabel) {
            threatLabelEl.textContent = threatLabel;
            flashValue(threatLabelEl);
        }
        threatLabelEl.style.color = statusColor;
    }

    const litCount = Math.min(5, Math.max(1, Math.ceil(STATE.risk / 20)));
    document.querySelectorAll("#threatSegments .segment").forEach((seg, i) => {
        if (i < litCount) {
            seg.style.background = statusColor;
            seg.style.borderColor = statusColor;
        } else {
            seg.style.background = "rgba(168,85,247,0.08)";
            seg.style.borderColor = "rgba(168,85,247,0.15)";
        }
    });

    // Confidence
    const confVal = (STATE.confidence || 0).toFixed(1) + "%";
    document.getElementById("confidence").textContent = confVal;
    document.getElementById("detectionConf").textContent = confVal;
    if (document.getElementById("heroAccuracy")) {
        document.getElementById("heroAccuracy").textContent = confVal;
    }

    // Prediction Box Mapping
    const rawEvent = (STATE.event || "").replace("Detected: ", "");
    document.getElementById("currentStage").textContent = rawEvent || "Benign";
    document.getElementById("predictedStage").textContent = STATE.next_attack || "Benign";
    
    const probPct = Math.round(STATE.confidence || 0);
    animateValue(document.getElementById("probValue"), probPct, { suffix: "%", duration: 400 });
    animateValue(document.getElementById("probPercentage"), probPct, { suffix: "%", duration: 400 });
    document.getElementById("probBar").style.width = probPct + "%";
    document.getElementById("predictedTime").textContent = `${probPct}% likely`;

    document.getElementById("forecastText").textContent = `Mitre Tactic: ${STATE.mitre || "Reconnaissance"} — Threat Level: ${threatLabel}`;

    // Quick Stats & MITRE
    animateValue(document.getElementById("anomalyScore"), STATE.anomaly || 0, { decimals: 1, duration: 400 });
    document.getElementById("mitreActiveStage").textContent = STATE.mitre || "Reconnaissance";
    document.getElementById("lastUpdate").textContent = formatTime();

    // Highlight MITRE Table Row
    const activeMitre = (STATE.mitre || "").toLowerCase();
    document.querySelectorAll("#mitreTableBody tr").forEach(row => {
        const tactic = (row.dataset.tactic || "").toLowerCase();
        row.classList.toggle("active-row", activeMitre.includes(tactic));
    });

    // Animate mini bars
    for (let i = 0; i < 5; i++) {
        const bar = document.getElementById(`bar${i}`);
        if (bar) {
            const h = Math.min(95, Math.max(15, (STATE.packets / 2) + rand(-5, 5)));
            bar.style.height = h + "%";
        }
    }

    updateTrafficChart();
}

function applyLiveNetworkData(data) {
    // Correctly handle both 'telemetry' and 'network_update' message formats from backend
    STATE.flows = Number(data.flows_sec ?? data.pps ?? data.flows ?? 0);
    STATE.packets = Number(data.packets_sec ?? data.pps ?? data.packets ?? 0);
    STATE.status = data.status === "online" ? "MONITORING" : (data.status ?? "MONITORING");

    if (data.event) STATE.event = data.event;
    if (data.risk !== undefined) STATE.risk = Number(data.risk);
    if (data.next_attack) STATE.next_attack = data.next_attack;
    if (data.mitre) STATE.mitre = data.mitre;
    if (data.confidence !== undefined) STATE.confidence = Number(data.confidence);
    if (data.anomaly !== undefined) STATE.anomaly = Number(data.anomaly);

    STATE.trafficHistory.push({ flows: STATE.flows, packets: STATE.packets });
    if (STATE.trafficHistory.length > 20) STATE.trafficHistory.shift();

    updateUI();
}

function fetchLatestForecast() {
    fetch("http://127.0.0.1:8000/api/predict/forecast")
        .then(res => res.json())
        .then(data => {
            if (data.predicted_next_stage) {
                STATE.next_attack = data.predicted_next_stage;
                STATE.confidence = (data.confidence || 0) * 100;

                // Map UNSW prediction classes to MITRE & Risk
                const cls = (data.predicted_next_stage || "").toLowerCase();
                if (cls.includes("normal") || cls.includes("benign")) {
                    STATE.mitre = "Reconnaissance";
                    STATE.risk = 10;
                    STATE.event = "Detected: Benign";
                } else if (cls.includes("reconnaissance") || cls.includes("fuzzers")) {
                    STATE.mitre = "Reconnaissance";
                    STATE.risk = 45;
                    STATE.event = "Detected: Network Reconnaissance";
                } else if (cls.includes("exploits") || cls.includes("shellcode")) {
                    STATE.mitre = "Initial Access";
                    STATE.risk = 85;
                    STATE.event = "Detected: Active Exploitation";
                } else {
                    STATE.mitre = "Command and Control";
                    STATE.risk = 65;
                    STATE.event = `Detected: ${data.predicted_next_stage}`;
                }
                updateUI();
            }
        })
        .catch(() => {});
}

function connectNetForeSightWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "127.0.0.1";
    const socket = new WebSocket(`${protocol}://${host}:8000/ws/alerts`);

    window.netForeSightSocket = socket;

    socket.addEventListener("open", () => {
        const sourceEl = document.getElementById("trafficSource");
        if (sourceEl) sourceEl.textContent = "LIVE MONITORING";
        const dot = document.getElementById("wsStatusDot");
        if (dot) dot.style.background = "#34d399";
    });

    socket.addEventListener("message", event => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "telemetry" || data.type === "network_update") {
                applyLiveNetworkData(data);
            }
        } catch (error) {
            console.error("Invalid WebSocket payload:", error);
        }
    });

    socket.addEventListener("close", () => {
        const sourceEl = document.getElementById("trafficSource");
        if (sourceEl) sourceEl.textContent = "BACKEND OFFLINE";
        const dot = document.getElementById("wsStatusDot");
        if (dot) dot.style.background = "#fb7185";
        setTimeout(connectNetForeSightWebSocket, 2000);
    });

    socket.addEventListener("error", () => socket.close());
}

buildTreeAndRender();
initCharts();
updateUI();
connectNetForeSightWebSocket();

// Periodically fetch AI forecast from FastAPI backend
setInterval(fetchLatestForecast, 2000);

/* ================= NAV SCROLLSPY ================= */
(function initScrollspy() {
    const targets = [
        { observeId: "dashboardAnchor", href: "#dashboard" },
        { observeId: "prediction", href: "#prediction" },
        { observeId: "mitre", href: "#mitre" },
        { observeId: "traffic-report", href: "#traffic-report" },
    ];
    const navLinks = Array.from(document.querySelectorAll('nav a.nav-link'));
    if (!navLinks.length) return;

    const setActive = (href) => {
        navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === href));
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const match = targets.find(t => t.observeId === entry.target.id);
                if (match) setActive(match.href);
            }
        });
    }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });

    targets.forEach(t => {
        const el = document.getElementById(t.observeId);
        if (el) observer.observe(el);
    });
})();

/* ================= HEADER SHRINK ON SCROLL ================= */
(function initHeaderShrink() {
    const header = document.querySelector('header.site-header');
    if (!header) return;
    const onScroll = () => header.classList.toggle('scrolled', window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
})();

/* ================= STAGGERED REVEAL ================= */
(function initScrollReveal() {
    if (prefersReducedMotion) return;
    const groups = document.querySelectorAll('main > .grid, #prediction');
    groups.forEach(group => {
        Array.from(group.children).forEach((child, i) => {
            child.classList.add('reveal');
            child.style.transitionDelay = `${Math.min(i, 4) * 70}ms`;
        });
    });
    const soloCards = document.querySelectorAll('main > .glass-strong, main > .glass');
    soloCards.forEach(card => card.classList.add('reveal'));

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });

    document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));
})();