import os
import time
import asyncio
import multiprocessing as mp
from queue import Empty, Full
from collections import deque, defaultdict

import joblib
import numpy as np
import pyshark

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Paths / configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "artifacts")

DEMO_MODEL_PATH = os.path.join(ARTIFACT_DIR, "demo_model.joblib")
DEMO_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "demo_label_encoder.joblib")
DEMO_METADATA_PATH = os.path.join(ARTIFACT_DIR, "demo_metadata.json")
DEMO_TEST_DATA_PATH = os.path.join(ARTIFACT_DIR, "demo_test_data.npz")

NETWORK_INTERFACE = os.getenv("NETFORESIGHT_INTERFACE", "4")
TSHARK_PATH = os.getenv(
    "TSHARK_PATH",
    r"C:\Program Files\Wireshark\tshark.exe",
)

FORECAST_HORIZON_SECONDS = 30
HISTORY_SECONDS = 60
REPLAY_HOLD_CALLS = 3


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="NetForesight V3 Demo Backend",
    version="3.0-demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ML artifacts
# ============================================================

demo_model = None
demo_label_encoder = None
demo_metadata = {}
demo_X = None
demo_y = None

# Deterministic replay state for the video/demo.
demo_order = []
demo_position = 0
demo_hold_counter = 0


# ============================================================
# Live capture state
# ============================================================

packet_queue = mp.Queue(maxsize=5000)
pyshark_process_handle = None

capture_stats = {
    "packets": 0,
    "bytes": 0,
    "flows": set(),
    "protocols": defaultdict(int),
}

recent_packet_times = deque(maxlen=20000)
recent_flow_events = deque(maxlen=20000)


# ============================================================
# Utility helpers
# ============================================================

def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def prune_recent_events():
    cutoff = time.time() - 1.0

    while recent_packet_times and recent_packet_times[0] < cutoff:
        recent_packet_times.popleft()

    while recent_flow_events and recent_flow_events[0][0] < cutoff:
        recent_flow_events.popleft()


def human_label(label):
    replacements = {
        "DDOS attack-HOIC": "DDoS - HOIC",
        "DDOS attack-LOIC-UDP": "DDoS - LOIC UDP",
        "DoS attacks-GoldenEye": "DoS - GoldenEye",
        "DoS attacks-Hulk": "DoS - Hulk",
        "DoS attacks-SlowHTTPTest": "DoS - SlowHTTPTest",
        "DoS attacks-Slowloris": "DoS - Slowloris",
        "FTP-BruteForce": "FTP Brute Force",
        "SSH-Bruteforce": "SSH Brute Force",
        "Benign": "Benign",
    }
    return replacements.get(label, label)


def mitre_tactic(label):
    if label == "Benign":
        return "Monitoring"

    lower = label.lower()

    if "brute" in lower:
        return "Credential Access"

    if "dos" in lower or "ddos" in lower:
        return "Impact"

    return "Network Attack"


def risk_for_prediction(label, benign_probability, confidence):
    if label == "Benign":
        return clamp(10 + (1.0 - benign_probability) * 25)

    lower = label.lower()
    anomaly = (1.0 - benign_probability) * 100.0

    if "brute" in lower:
        return clamp(max(65.0, anomaly))

    if "dos" in lower or "ddos" in lower:
        return clamp(max(80.0, anomaly))

    return clamp(max(60.0, anomaly, confidence * 100.0))


# ============================================================
# ML loading
# ============================================================

def load_demo_artifacts():
    global demo_model
    global demo_label_encoder
    global demo_metadata
    global demo_X
    global demo_y
    global demo_order

    print("\n" + "=" * 70)
    print("Loading NetForesight V3 demo model")
    print("=" * 70)

    required = [
        DEMO_MODEL_PATH,
        DEMO_ENCODER_PATH,
        DEMO_METADATA_PATH,
        DEMO_TEST_DATA_PATH,
    ]

    missing = [path for path in required if not os.path.exists(path)]

    if missing:
        raise RuntimeError(
            "Missing V3 demo artifacts:\n" + "\n".join(missing)
        )

    demo_model = joblib.load(DEMO_MODEL_PATH)
    demo_label_encoder = joblib.load(DEMO_ENCODER_PATH)

    with open(DEMO_METADATA_PATH, "r", encoding="utf-8") as file:
        import json
        demo_metadata = json.load(file)

    test_data = np.load(DEMO_TEST_DATA_PATH)
    demo_X = test_data["X"].astype(np.float32)
    demo_y = test_data["y"].astype(int)

    expected_features = int(demo_metadata.get("num_input_features", demo_X.shape[1]))

    if demo_X.ndim != 2 or demo_X.shape[1] != expected_features:
        raise RuntimeError(
            f"Demo data shape mismatch: got {demo_X.shape}, "
            f"expected second dimension {expected_features}."
        )

    if demo_X.shape[1] != int(getattr(demo_model, "n_features_in_", demo_X.shape[1])):
        raise RuntimeError(
            "Demo model input size does not match demo_test_data.npz."
        )

    # Build a deterministic replay order that deliberately includes
    # attack classes first, followed by benign samples.
    classes = list(demo_label_encoder.classes_)

    attack_class_ids = [
        idx for idx, name in enumerate(classes)
        if name != "Benign"
    ]

    per_class = []
    for class_id in attack_class_ids:
        indices = np.where(demo_y == class_id)[0]
        if len(indices):
            per_class.append(int(indices[0]))

    benign_indices = np.where(
        demo_y == demo_label_encoder.transform(["Benign"])[0]
    )[0]

    if len(benign_indices):
        per_class.append(int(benign_indices[0]))

    # Add the first few samples as fallback if the class-based order
    # is unexpectedly short.
    if not per_class:
        per_class = list(range(min(len(demo_X), 20)))

    demo_order = per_class

    print(f"[✔] Demo model loaded: {DEMO_MODEL_PATH}")
    print(f"[✔] Classes: {classes}")
    print(f"[✔] Demo samples: {len(demo_X):,}")
    print(f"[✔] Input features: {demo_X.shape[1]}")
    print(f"[✔] History: {HISTORY_SECONDS}s")
    print(f"[✔] Forecast horizon: {FORECAST_HORIZON_SECONDS}s")


# ============================================================
# PyShark worker
# ============================================================

def pyshark_worker_process(packet_queue, interface, tshark_path):
    print(f"[+] Starting PyShark worker on interface: {interface}")

    try:
        kwargs = {
            "interface": interface,
            "bpf_filter": "ip and (tcp or udp)",
            "use_json": True,
            "include_raw": False,
        }

        if tshark_path and os.path.exists(tshark_path):
            kwargs["tshark_path"] = tshark_path

        capture = pyshark.LiveCapture(**kwargs)
        print("[✔] PyShark capture started.")

        for packet in capture.sniff_continuously():
            try:
                timestamp = time.time()

                length = safe_int(getattr(packet, "length", 0))

                src_ip = ""
                dst_ip = ""

                if hasattr(packet, "ip"):
                    src_ip = str(getattr(packet.ip, "src", ""))
                    dst_ip = str(getattr(packet.ip, "dst", ""))
                elif hasattr(packet, "ipv6"):
                    src_ip = str(getattr(packet.ipv6, "src", ""))
                    dst_ip = str(getattr(packet.ipv6, "dst", ""))

                src_port = 0
                dst_port = 0
                protocol = "OTHER"

                if hasattr(packet, "tcp"):
                    src_port = safe_int(getattr(packet.tcp, "srcport", 0))
                    dst_port = safe_int(getattr(packet.tcp, "dstport", 0))
                    protocol = "tcp"
                elif hasattr(packet, "udp"):
                    src_port = safe_int(getattr(packet.udp, "srcport", 0))
                    dst_port = safe_int(getattr(packet.udp, "dstport", 0))
                    protocol = "udp"

                flow_key = (
                    src_ip,
                    dst_ip,
                    src_port,
                    dst_port,
                    protocol,
                )

                packet_data = (
                    timestamp,
                    length,
                    flow_key,
                    protocol,
                )

                try:
                    packet_queue.put_nowait(packet_data)
                except Full:
                    pass

            except Exception:
                continue

    except Exception as exc:
        print(f"[-] PyShark worker crashed: {exc}")


# ============================================================
# Queue / telemetry
# ============================================================

def drain_pyshark_queue():
    while True:
        try:
            timestamp, length, flow_key, protocol = packet_queue.get_nowait()

            capture_stats["packets"] += 1
            capture_stats["bytes"] += length
            capture_stats["flows"].add(flow_key)
            capture_stats["protocols"][protocol] += 1

            recent_packet_times.append(timestamp)
            recent_flow_events.append((timestamp, flow_key))

        except Empty:
            break
        except Exception:
            break

    prune_recent_events()


def live_rates():
    prune_recent_events()
    packets_sec = len(recent_packet_times)
    flows_sec = len({event[1] for event in recent_flow_events})
    return packets_sec, flows_sec


# ============================================================
# Demo forecast
# ============================================================

def next_demo_index():
    global demo_position
    global demo_hold_counter

    if not demo_order:
        return 0

    if demo_hold_counter < REPLAY_HOLD_CALLS:
        demo_hold_counter += 1
    else:
        demo_hold_counter = 1
        demo_position = (demo_position + 1) % len(demo_order)

    return demo_order[demo_position]


def make_forecast(index):
    sample = demo_X[index:index + 1]

    probabilities = demo_model.predict_proba(sample)[0]
    model_classes = [int(value) for value in demo_model.classes_]

    ranked = sorted(
        zip(model_classes, probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    ranked_predictions = []

    for class_id, probability in ranked:
        class_name = str(
            demo_label_encoder.inverse_transform([class_id])[0]
        )
        ranked_predictions.append(
            {
                "stage": human_label(class_name),
                "label": class_name,
                "probability": float(probability),
                "percentage": round(float(probability) * 100.0, 2),
            }
        )

    top_class_id, top_probability = ranked[0]
    predicted_label = str(
        demo_label_encoder.inverse_transform([top_class_id])[0]
    )

    benign_id = int(
        demo_label_encoder.transform(["Benign"])[0]
    )

    benign_probability = 0.0
    for class_id, probability in zip(model_classes, probabilities):
        if class_id == benign_id:
            benign_probability = float(probability)
            break

    confidence = float(top_probability)
    risk = risk_for_prediction(
        predicted_label,
        benign_probability,
        confidence,
    )

    anomaly = clamp((1.0 - benign_probability) * 100.0)
    tactic = mitre_tactic(predicted_label)

    return {
        "predicted_next_stage": human_label(predicted_label),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "raw_confidence": confidence,
        "class_probabilities": {
            human_label(str(demo_label_encoder.inverse_transform([class_id])[0])): float(probability)
            for class_id, probability in zip(model_classes, probabilities)
        },
        "ranked_predictions": ranked_predictions,
        "window_packet_count": 0,
        "sequence_length": 6,
        "feature_count": int(demo_X.shape[1]),
        "current_stage": "Observed Network Activity",
        "risk": risk,
        "anomaly": anomaly,
        "mitre": tactic,
        "event": (
            "Forecast: Normal Traffic"
            if predicted_label == "Benign"
            else f"Forecast: {human_label(predicted_label)}"
        ),
        "forecast_horizon_seconds": FORECAST_HORIZON_SECONDS,
        "history_seconds": HISTORY_SECONDS,
        "source": "CIC-IDS-2018 demo replay",
        "demo_sample_index": index,
    }


# ============================================================
# Lifecycle
# ============================================================

@app.on_event("startup")
def startup_event():
    global pyshark_process_handle

    print("\n" + "=" * 70)
    print("NetForesight V3 demo backend starting...")
    print("=" * 70)

    try:
        load_demo_artifacts()
    except Exception as exc:
        print(f"[-] ML initialization failed:\n{exc}")
        return

    if os.path.exists(TSHARK_PATH):
        try:
            pyshark_process_handle = mp.Process(
                target=pyshark_worker_process,
                args=(packet_queue, NETWORK_INTERFACE, TSHARK_PATH),
                daemon=True,
            )
            pyshark_process_handle.start()
            print("[✔] PyShark worker started.")
        except Exception as exc:
            print(f"[!] Failed to start PyShark: {exc}")
    else:
        print(f"[!] tshark.exe not found at: {TSHARK_PATH}")
        print("[!] ML demo replay will still work.")

    print("[✔] NetForesight V3 demo backend is ready.")


@app.on_event("shutdown")
def shutdown_event():
    global pyshark_process_handle

    if pyshark_process_handle and pyshark_process_handle.is_alive():
        print("[+] Stopping PyShark worker...")
        pyshark_process_handle.terminate()
        pyshark_process_handle.join(timeout=2)
        print("[✔] PyShark worker stopped.")


# ============================================================
# WebSocket telemetry
# ============================================================

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            drain_pyshark_queue()
            packets_sec, flows_sec = live_rates()

            await websocket.send_json(
                {
                    "type": "telemetry",
                    "status": "online",
                    "packets": capture_stats["packets"],
                    "bytes": capture_stats["bytes"],
                    "flows": len(capture_stats["flows"]),
                    "packets_sec": packets_sec,
                    "flows_sec": flows_sec,
                    "protocols": dict(capture_stats["protocols"]),
                    "source": "live telemetry",
                }
            )

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ============================================================
# REST endpoints
# ============================================================

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "NetForesight V3",
        "version": "3.0-demo",
        "model": "Random Forest - CIC-IDS-2018 temporal demo",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "demo_model_loaded": demo_model is not None,
        "label_encoder_loaded": demo_label_encoder is not None,
        "feature_count": int(demo_X.shape[1]) if demo_X is not None else 0,
        "capture_worker_running": (
            pyshark_process_handle is not None
            and pyshark_process_handle.is_alive()
        ),
        "source": "CIC-IDS-2018 demo replay",
    }


@app.get("/api/capture/stats")
def get_capture_stats():
    drain_pyshark_queue()
    packets_sec, flows_sec = live_rates()

    return {
        "packets": capture_stats["packets"],
        "bytes": capture_stats["bytes"],
        "flows": len(capture_stats["flows"]),
        "packets_sec": packets_sec,
        "flows_sec": flows_sec,
        "protocols": dict(capture_stats["protocols"]),
        "source": "live telemetry",
    }


@app.get("/api/predict/forecast")
def forecast_next_stage():
    if demo_model is None or demo_label_encoder is None or demo_X is None:
        raise HTTPException(
            status_code=500,
            detail="V3 demo ML artifacts are not loaded.",
        )

    try:
        index = next_demo_index()
        result = make_forecast(index)

        # Fill the live packet count into the response without using
        # the live traffic as model input.
        drain_pyshark_queue()
        result["window_packet_count"] = len(recent_packet_times)

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        )


@app.get("/api/model/info")
def model_info():
    return {
        "demo_model_loaded": demo_model is not None,
        "model_type": "RandomForestClassifier",
        "classes": (
            list(demo_label_encoder.classes_)
            if demo_label_encoder is not None
            else []
        ),
        "feature_count": int(demo_X.shape[1]) if demo_X is not None else 0,
        "history_seconds": HISTORY_SECONDS,
        "forecast_horizon_seconds": FORECAST_HORIZON_SECONDS,
        "bucket_seconds": demo_metadata.get("bucket_seconds"),
        "source": "CIC-IDS-2018 demo replay",
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn

    mp.freeze_support()

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
