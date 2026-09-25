import os
import time
import asyncio
import multiprocessing as mp

from queue import Empty, Full
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import pyshark

from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "feature_scaler.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR,
    "feature_names.pkl"
)

TRANSFORMER_PATH = os.path.join(
    BASE_DIR,
    "transformer_forecaster.pt"
)

XGB_PATH = os.path.join(
    BASE_DIR,
    "xgboost_model.pkl"
)

SEQUENCE_LENGTH = 5

# You can override these from Git Bash:
#
# export NETFORESIGHT_INTERFACE=4
# export TSHARK_PATH="C:/Program Files/Wireshark/tshark.exe"
#
NETWORK_INTERFACE = os.getenv(
    "NETFORESIGHT_INTERFACE",
    "4"
)

TSHARK_PATH = os.getenv(
    "TSHARK_PATH",
    r"C:\Program Files\Wireshark\tshark.exe"
)

WINDOW_SECONDS = 2.0


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="NetForesight V2 Backend",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global ML artifacts
# ============================================================

label_encoder = None
feature_scaler = None
feature_names = None

transformer_model = None
xgboost_model = None

categorical_encoders = {}


# ============================================================
# Runtime state
# ============================================================

packet_queue = mp.Queue(maxsize=5000)

pyshark_process_handle = None

capture_stats = {
    "packets": 0,
    "bytes": 0,
    "flows": set(),
    "protocols": defaultdict(int),
}


# ============================================================
# Utility
# ============================================================

def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


# ============================================================
# Categorical encoders
# ============================================================

def load_categorical_encoders():
    """
    Reconstruct the categorical encoders used during training.

    The training pipeline uses the combined UNSW train/test data
    to establish the possible values for proto/service/state.
    """

    global categorical_encoders

    categorical_encoders = {}

    train_path = os.path.join(
        DATA_DIR,
        "UNSW_NB15_training-set.csv"
    )

    test_path = os.path.join(
        DATA_DIR,
        "UNSW_NB15_testing-set.csv"
    )

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(
            "[!] UNSW CSV files not found."
        )

        return

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        combined = pd.concat(
            [train_df, test_df],
            ignore_index=True
        )

        for column in ["proto", "service", "state"]:

            if column not in combined.columns:
                continue

            encoder = LabelEncoder()

            encoder.fit(
                combined[column]
                .astype(str)
                .values
            )

            categorical_encoders[column] = encoder

            print(
                f"[✔] Encoder loaded: {column} "
                f"({len(encoder.classes_)} values)"
            )

    except Exception as exc:

        print(
            f"[!] Could not reconstruct categorical encoders: {exc}"
        )


def encode_categorical(column, value):
    """
    Convert a live categorical value into the same numerical
    representation used during training.
    """

    encoder = categorical_encoders.get(column)

    if encoder is None:
        return 0.0

    value = str(value)

    try:
        return float(
            encoder.transform([value])[0]
        )

    except ValueError:
        return 0.0


# ============================================================
# PyShark worker
# ============================================================

def pyshark_worker_process(
    packet_queue,
    interface,
    tshark_path
):
    """
    Dedicated process for PyShark.

    Keeping packet capture outside FastAPI prevents tshark
    processing from blocking the API/event loop.
    """

    print(
        f"[+] Starting PyShark worker on interface: {interface}"
    )

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

                length = safe_int(
                    getattr(
                        packet,
                        "length",
                        0
                    )
                )

                src_ip = ""
                dst_ip = ""

                if hasattr(packet, "ip"):

                    src_ip = str(
                        getattr(
                            packet.ip,
                            "src",
                            ""
                        )
                    )

                    dst_ip = str(
                        getattr(
                            packet.ip,
                            "dst",
                            ""
                        )
                    )

                elif hasattr(packet, "ipv6"):

                    src_ip = str(
                        getattr(
                            packet.ipv6,
                            "src",
                            ""
                        )
                    )

                    dst_ip = str(
                        getattr(
                            packet.ipv6,
                            "dst",
                            ""
                        )
                    )

                src_port = 0
                dst_port = 0

                protocol = "OTHER"

                is_tcp = 0.0
                is_udp = 0.0

                if hasattr(packet, "tcp"):

                    src_port = safe_int(
                        getattr(
                            packet.tcp,
                            "srcport",
                            0
                        )
                    )

                    dst_port = safe_int(
                        getattr(
                            packet.tcp,
                            "dstport",
                            0
                        )
                    )

                    protocol = "tcp"
                    is_tcp = 1.0

                elif hasattr(packet, "udp"):

                    src_port = safe_int(
                        getattr(
                            packet.udp,
                            "srcport",
                            0
                        )
                    )

                    dst_port = safe_int(
                        getattr(
                            packet.udp,
                            "dstport",
                            0
                        )
                    )

                    protocol = "udp"
                    is_udp = 1.0

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
                    src_ip,
                    dst_ip,
                    src_port,
                    dst_port,
                    protocol,
                    is_tcp,
                    is_udp,
                    flow_key,
                )

                try:

                    packet_queue.put_nowait(
                        packet_data
                    )

                except Full:

                    # Dropping packets is preferable to allowing
                    # the queue to grow indefinitely.
                    pass

            except Exception:
                continue

    except Exception as exc:

        print(
            f"[-] PyShark worker crashed: {exc}"
        )


# ============================================================
# Realtime feature aggregator
# ============================================================

class RealtimeFlowAggregator:

    def __init__(
        self,
        window_sec=2.0,
        seq_len=5
    ):

        self.window_sec = window_sec
        self.seq_len = seq_len

        self.packet_history = deque()

        self.flow_sequence = deque(
            maxlen=seq_len
        )

        self.smoothed_confidence = 0.0

        self.alpha = 0.3

    # --------------------------------------------------------
    # Add packet
    # --------------------------------------------------------

    def add_packet(
        self,
        packet_data
    ):

        self.packet_history.append(
            packet_data
        )

    # --------------------------------------------------------
    # Remove old packets
    # --------------------------------------------------------

    def clean_old_packets(self):

        now = time.time()

        cutoff = (
            now -
            self.window_sec
        )

        while (
            self.packet_history
            and
            self.packet_history[0][0] < cutoff
        ):

            self.packet_history.popleft()

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def extract_unsw_features(self):

        self.clean_old_packets()

        packets = list(
            self.packet_history
        )

        now = time.time()

        # ----------------------------------------------------
        # Initialize all UNSW-NB15 features
        # ----------------------------------------------------

        features = {
            "dur": 0.0,

            "proto": 0.0,
            "service": 0.0,
            "state": 0.0,

            "spkts": 0.0,
            "dpkts": 0.0,

            "sbytes": 0.0,
            "dbytes": 0.0,

            "rate": 0.0,

            "sttl": 0.0,
            "dttl": 0.0,

            "sload": 0.0,
            "dload": 0.0,

            "sloss": 0.0,
            "dloss": 0.0,

            "sinpkt": 0.0,
            "dinpkt": 0.0,

            "sjit": 0.0,
            "djit": 0.0,

            "swin": 0.0,
            "stcpb": 0.0,
            "dtcpb": 0.0,
            "dwin": 0.0,

            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,

            "smean": 0.0,
            "dmean": 0.0,

            "trans_depth": 0.0,

            "response_body_len": 0.0,

            "ct_srv_src": 0.0,
            "ct_state_ttl": 0.0,
            "ct_dst_ltm": 0.0,
            "ct_src_dport_ltm": 0.0,
            "ct_dst_sport_ltm": 0.0,
            "ct_dst_src_ltm": 0.0,

            "is_ftp_login": 0.0,
            "ct_ftp_cmd": 0.0,
            "ct_flw_http_mthd": 0.0,

            "ct_src_ltm": 0.0,
            "ct_srv_dst": 0.0,

            "is_sm_ips_ports": 0.0,
        }

        # ----------------------------------------------------
        # No packets
        # ----------------------------------------------------

        if not packets:

            return features

        # ----------------------------------------------------
        # Basic packet statistics
        # ----------------------------------------------------

        first_time = packets[0][0]
        last_time = packets[-1][0]

        duration = max(
            last_time - first_time,
            0.001
        )

        features["dur"] = duration

        total_packets = len(packets)

        features["spkts"] = float(
            total_packets
        )

        features["sbytes"] = float(
            sum(
                max(0, packet[1])
                for packet in packets
            )
        )

        features["rate"] = (
            total_packets /
            duration
        )

        features["sload"] = (
            features["sbytes"] *
            8.0 /
            duration
        )

        features["smean"] = (
            features["sbytes"] /
            max(total_packets, 1)
        )

        # ----------------------------------------------------
        # Protocol information
        # ----------------------------------------------------

        tcp_packets = sum(
            1
            for packet in packets
            if packet[7] == 1.0
        )

        udp_packets = sum(
            1
            for packet in packets
            if packet[8] == 1.0
        )

        protocol = "tcp"

        if udp_packets > tcp_packets:
            protocol = "udp"

        features["proto"] = encode_categorical(
            "proto",
            protocol
        )

        # ----------------------------------------------------
        # Approximate service using destination port
        # ----------------------------------------------------

        latest_dst_port = packets[-1][5]

        service_map = {
            20: "ftp",
            21: "ftp",
            22: "ssh",
            23: "telnet",
            25: "smtp",
            53: "dns",
            80: "http",
            110: "pop3",
            143: "imap4",
            443: "http",
            445: "-",
            993: "imap4",
            995: "pop3",
        }

        service = service_map.get(
            latest_dst_port,
            "-"
        )

        features["service"] = encode_categorical(
            "service",
            service
        )

        # ----------------------------------------------------
        # Connection state approximation
        # ----------------------------------------------------

        state = "INT"

        if tcp_packets > 0:
            state = "CON"

        features["state"] = encode_categorical(
            "state",
            state
        )

        # ----------------------------------------------------
        # Packet timing
        # ----------------------------------------------------

        timestamps = [
            packet[0]
            for packet in packets
        ]

        if len(timestamps) >= 2:

            intervals = np.diff(
                timestamps
            )

            avg_interval = float(
                np.mean(intervals)
            )

            features["sinpkt"] = (
                avg_interval
            )

            features["sjit"] = float(
                np.std(intervals)
            )

        # ----------------------------------------------------
        # TCP approximations
        # ----------------------------------------------------

        if tcp_packets > 0:

            features["swin"] = 0.0

            features["dwin"] = 0.0

            features["tcprtt"] = 0.0

            features["synack"] = 0.0

            features["ackdat"] = 0.0

        # ----------------------------------------------------
        # Same IP/port indicator
        # ----------------------------------------------------

        same_ip_port = all(
            (
                packet[2] == packets[0][2]
                and
                packet[3] == packets[0][3]
            )
            for packet in packets
        )

        features["is_sm_ips_ports"] = (
            1.0
            if same_ip_port
            else 0.0
        )

        # ----------------------------------------------------
        # Add small contextual counts
        # ----------------------------------------------------

        unique_dst_ports = len(
            set(
                packet[5]
                for packet in packets
            )
        )

        unique_src_ports = len(
            set(
                packet[4]
                for packet in packets
            )
        )

        unique_destinations = len(
            set(
                packet[3]
                for packet in packets
            )
        )

        features["ct_dst_ltm"] = float(
            unique_destinations
        )

        features["ct_src_ltm"] = float(
            unique_src_ports
        )

        features["ct_src_dport_ltm"] = float(
            unique_dst_ports
        )

        features["ct_dst_src_ltm"] = float(
            unique_destinations
        )

        features["ct_srv_dst"] = float(
            unique_destinations
        )

        features["ct_srv_src"] = float(
            unique_destinations
        )

        # ----------------------------------------------------
        # Keep everything finite
        # ----------------------------------------------------

        for key in features:

            value = safe_float(
                features[key]
            )

            features[key] = value

        return features

    # --------------------------------------------------------
    # Create model vector
    # --------------------------------------------------------

    def create_feature_vector(
        self,
        feature_names
    ):

        features = (
            self.extract_unsw_features()
        )

        vector = []

        for name in feature_names:

            vector.append(
                features.get(
                    name,
                    0.0
                )
            )

        return np.array(
            vector,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Sequence generation
    # --------------------------------------------------------

    def create_sequence(
        self,
        feature_names
    ):

        feature_vector = (
            self.create_feature_vector(
                feature_names
            )
        )

        self.flow_sequence.append(
            feature_vector
        )

        sequence = list(
            self.flow_sequence
        )

        # Pad beginning of sequence.
        while len(sequence) < self.seq_len:

            sequence.insert(
                0,
                np.zeros(
                    len(feature_names),
                    dtype=np.float32
                )
            )

        return np.array(
            sequence,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Confidence smoothing
    # --------------------------------------------------------

    def smooth_confidence(
        self,
        confidence
    ):

        self.smoothed_confidence = (
            self.alpha * confidence
            +
            (1.0 - self.alpha)
            * self.smoothed_confidence
        )

        return self.smoothed_confidence


aggregator = RealtimeFlowAggregator(
    window_sec=WINDOW_SECONDS,
    seq_len=SEQUENCE_LENGTH
)


# ============================================================
# Queue draining
# ============================================================

def drain_pyshark_queue():

    while True:

        try:

            packet = (
                packet_queue.get_nowait()
            )

            (
                timestamp,
                length,
                src_ip,
                dst_ip,
                src_port,
                dst_port,
                protocol,
                is_tcp,
                is_udp,
                flow_key,
            ) = packet

            capture_stats["packets"] += 1

            capture_stats["bytes"] += (
                length
            )

            capture_stats[
                "protocols"
            ][protocol] += 1

            capture_stats[
                "flows"
            ].add(flow_key)

            aggregator.add_packet(
                packet
            )

        except Empty:

            break

        except Exception:

            break


# ============================================================
# Transformer
# ============================================================

class AttackForecasterTransformer(
    nn.Module
):

    def __init__(
        self,
        feature_dim,
        seq_len=5,
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.15,
    ):

        super().__init__()

        self.embedding = nn.Linear(
            feature_dim,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        x = self.embedding(x)

        x = (
            x +
            self.position_embedding
        )

        x = self.transformer(x)

        x = self.norm(x)

        # Final timestep represents the most recent
        # observed network state.
        x = x[:, -1, :]

        return self.classifier(x)


# ============================================================
# Load ML artifacts
# ============================================================

def load_ml_artifacts():

    global label_encoder
    global feature_scaler
    global feature_names
    global transformer_model
    global xgboost_model

    print("\n" + "=" * 70)
    print("Loading NetForesight V2 ML artifacts")
    print("=" * 70)

    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    try:

        label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        print(
            f"[✔] Label encoder loaded "
            f"({len(label_encoder.classes_)} classes)"
        )

        print(
            f"    Classes: "
            f"{list(label_encoder.classes_)}"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load label encoder: {exc}"
        )

    # --------------------------------------------------------
    # Feature scaler
    # --------------------------------------------------------

    try:

        feature_scaler = joblib.load(
            SCALER_PATH
        )

        print(
            "[✔] Feature scaler loaded."
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load feature scaler: {exc}"
        )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    try:

        feature_names = joblib.load(
            FEATURE_NAMES_PATH
        )

        print(
            f"[✔] Feature names loaded "
            f"({len(feature_names)} features)"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load feature names: {exc}"
        )

    # --------------------------------------------------------
    # Categorical encoders
    # --------------------------------------------------------

    load_categorical_encoders()

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    try:

        if os.path.exists(
            XGB_PATH
        ):

            xgboost_model = joblib.load(
                XGB_PATH
            )

            print(
                "[✔] XGBoost model loaded."
            )

        else:

            print(
                "[!] XGBoost model not found."
            )

    except Exception as exc:

        print(
            f"[!] XGBoost could not be loaded: {exc}"
        )

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            TRANSFORMER_PATH,
            map_location="cpu",
            weights_only=False
        )

        # ----------------------------------------------------
        # New V2 checkpoint format
        # ----------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ) and "state_dict" in checkpoint:

            state_dict = (
                checkpoint["state_dict"]
            )

            model_feature_dim = (
                checkpoint.get(
                    "feature_dim",
                    len(feature_names)
                )
            )

            model_seq_len = (
                checkpoint.get(
                    "sequence_length",
                    SEQUENCE_LENGTH
                )
            )

            model_num_classes = (
                checkpoint.get(
                    "num_classes",
                    len(label_encoder.classes_)
                )
            )

            d_model = checkpoint.get(
                "d_model",
                64
            )

            nhead = checkpoint.get(
                "nhead",
                4
            )

            num_layers = checkpoint.get(
                "num_layers",
                2
            )

            dropout = checkpoint.get(
                "dropout",
                0.15
            )

        # ----------------------------------------------------
        # Raw state_dict fallback
        # ----------------------------------------------------

        else:

            state_dict = checkpoint

            model_feature_dim = (
                len(feature_names)
            )

            model_seq_len = (
                SEQUENCE_LENGTH
            )

            model_num_classes = (
                len(label_encoder.classes_)
            )

            d_model = 64
            nhead = 4
            num_layers = 2
            dropout = 0.15

        # ----------------------------------------------------
        # Verify feature dimensions
        # ----------------------------------------------------

        if model_feature_dim != len(
            feature_names
        ):

            raise RuntimeError(
                "Transformer feature dimension "
                f"({model_feature_dim}) does not match "
                f"feature_names ({len(feature_names)})."
            )

        # ----------------------------------------------------
        # Create architecture
        # ----------------------------------------------------

        transformer_model = (
            AttackForecasterTransformer(
                feature_dim=model_feature_dim,
                seq_len=model_seq_len,
                num_classes=model_num_classes,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
            )
        )

        transformer_model.load_state_dict(
            state_dict
        )

        transformer_model.eval()

        print(
            "[✔] Transformer loaded successfully."
        )

        print(
            f"    Features : {model_feature_dim}"
        )

        print(
            f"    Sequence : {model_seq_len}"
        )

        print(
            f"    Classes  : {model_num_classes}"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load Transformer: {exc}"
        )


# ============================================================
# Start PyShark
# ============================================================

def start_pyshark():

    global pyshark_process_handle

    if not os.path.exists(
        TSHARK_PATH
    ):

        print(
            "[!] tshark.exe was not found at:"
        )

        print(
            f"    {TSHARK_PATH}"
        )

        print(
            "[!] Packet capture will not start."
        )

        return

    try:

        pyshark_process_handle = (
            mp.Process(
                target=pyshark_worker_process,
                args=(
                    packet_queue,
                    NETWORK_INTERFACE,
                    TSHARK_PATH,
                ),
                daemon=True,
            )
        )

        pyshark_process_handle.start()

        print(
            "[✔] PyShark worker started."
        )

        print(
            f"    Interface: {NETWORK_INTERFACE}"
        )

    except Exception as exc:

        print(
            f"[!] Failed to start PyShark: {exc}"
        )


# ============================================================
# FastAPI lifecycle
# ============================================================

@app.on_event("startup")
def startup_event():

    print("\n" + "=" * 70)
    print("NetForesight V2 starting...")
    print("=" * 70)

    try:

        load_ml_artifacts()

    except Exception as exc:

        print(
            f"\n[-] ML initialization failed:\n{exc}"
        )

        return

    start_pyshark()

    print(
        "\n[✔] NetForesight V2 backend is ready."
    )


@app.on_event("shutdown")
def shutdown_event():

    global pyshark_process_handle

    if (
        pyshark_process_handle
        and
        pyshark_process_handle.is_alive()
    ):

        print(
            "[+] Stopping PyShark worker..."
        )

        pyshark_process_handle.terminate()

        pyshark_process_handle.join(
            timeout=2
        )

        print(
            "[✔] PyShark worker stopped."
        )


# ============================================================
# WebSocket manager
# ============================================================

class ConnectionManager:

    def __init__(self):

        self.active_connections = []

    async def connect(
        self,
        websocket
    ):

        await websocket.accept()

        self.active_connections.append(
            websocket
        )

    def disconnect(
        self,
        websocket
    ):

        if websocket in self.active_connections:

            self.active_connections.remove(
                websocket
            )

    async def broadcast(
        self,
        message
    ):

        for connection in list(
            self.active_connections
        ):

            try:

                await connection.send_json(
                    message
                )

            except Exception:

                self.disconnect(
                    connection
                )


manager = ConnectionManager()


# ============================================================
# WebSocket telemetry
# ============================================================

@app.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket
):

    await manager.connect(
        websocket
    )

    try:

        while True:

            drain_pyshark_queue()

            await websocket.send_json(
                {
                    "type": "telemetry",
                    "status": "online",
                    "packets": capture_stats[
                        "packets"
                    ],
                    "bytes": capture_stats[
                        "bytes"
                    ],
                    "flows": len(
                        capture_stats[
                            "flows"
                        ]
                    ),
                    "protocols": dict(
                        capture_stats[
                            "protocols"
                        ]
                    ),
                }
            )

            await asyncio.sleep(
                1
            )

    except WebSocketDisconnect:

        manager.disconnect(
            websocket
        )

    except Exception:

        manager.disconnect(
            websocket
        )


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def read_root():

    return {
        "status": "online",
        "service": "NetForesight V2",
        "version": "4.0",
        "model": (
            "Transformer + XGBoost"
            if xgboost_model is not None
            else "Transformer"
        ),
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",

        "transformer_loaded":
            transformer_model is not None,

        "xgboost_loaded":
            xgboost_model is not None,

        "label_encoder_loaded":
            label_encoder is not None,

        "scaler_loaded":
            feature_scaler is not None,

        "feature_count":
            len(feature_names)
            if feature_names is not None
            else 0,

        "capture_worker_running":
            (
                pyshark_process_handle is not None
                and
                pyshark_process_handle.is_alive()
            ),
    }


# ============================================================
# Capture statistics
# ============================================================

@app.get("/api/capture/stats")
def get_capture_stats():

    drain_pyshark_queue()

    return {
        "packets":
            capture_stats["packets"],

        "bytes":
            capture_stats["bytes"],

        "flows":
            len(
                capture_stats["flows"]
            ),

        "protocols":
            dict(
                capture_stats[
                    "protocols"
                ]
            ),

        "buffer_size":
            len(
                aggregator.packet_history
            ),
    }


# ============================================================
# Transformer prediction
# ============================================================

@app.get("/api/predict/forecast")
def forecast_next_stage():

    if (
        transformer_model is None
        or
        label_encoder is None
        or
        feature_scaler is None
        or
        feature_names is None
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Transformer or preprocessing "
                "artifacts are not loaded."
            ),
        )

    # --------------------------------------------------------
    # Bring newest packets into memory
    # --------------------------------------------------------

    drain_pyshark_queue()

    try:

        # ----------------------------------------------------
        # Create live 5-step sequence
        # ----------------------------------------------------

        raw_sequence = (
            aggregator.create_sequence(
                feature_names
            )
        )

        # ----------------------------------------------------
        # Scale exactly like training
        # ----------------------------------------------------

        scaled_sequence = (
            feature_scaler.transform(
                raw_sequence
            )
        )

        tensor_input = torch.tensor(
            scaled_sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Transformer inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = (
                transformer_model(
                    tensor_input
                )
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()[0]
            )

        # ----------------------------------------------------
        # Rank every possible class
        # ----------------------------------------------------

        ranked_indices = np.argsort(
            probabilities
        )[::-1]

        ranked_predictions = []

        for index in ranked_indices:

            class_name = (
                label_encoder
                .inverse_transform(
                    [int(index)]
                )[0]
            )

            probability = float(
                probabilities[index]
            )

            ranked_predictions.append(
                {
                    "stage": class_name,
                    "probability": probability,
                    "percentage": round(
                        probability * 100,
                        2
                    ),
                }
            )

        # ----------------------------------------------------
        # Top prediction
        # ----------------------------------------------------

        top_index = int(
            ranked_indices[0]
        )

        predicted_class = (
            label_encoder
            .inverse_transform(
                [top_index]
            )[0]
        )

        raw_confidence = float(
            probabilities[top_index]
        )

        smoothed_confidence = (
            aggregator.smooth_confidence(
                raw_confidence
            )
        )

        # ----------------------------------------------------
        # Probability dictionary
        # ----------------------------------------------------

        class_probabilities = {}

        for index, class_name in enumerate(
            label_encoder.classes_
        ):

            class_probabilities[
                class_name
            ] = float(
                probabilities[index]
            )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "predicted_next_stage":
                predicted_class,

            "confidence":
                float(
                    smoothed_confidence
                ),

            "raw_confidence":
                raw_confidence,

            "class_probabilities":
                class_probabilities,

            "ranked_predictions":
                ranked_predictions,

            "window_packet_count":
                len(
                    aggregator.packet_history
                ),

            "sequence_length":
                SEQUENCE_LENGTH,

            "feature_count":
                len(feature_names),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference error: {str(exc)}"
            ),
        )


# ============================================================
# Manual prediction endpoint
# ============================================================

@app.get("/api/model/info")
def model_info():

    return {

        "transformer_loaded":
            transformer_model is not None,

        "xgboost_loaded":
            xgboost_model is not None,

        "classes":
            (
                list(
                    label_encoder.classes_
                )
                if label_encoder is not None
                else []
            ),

        "feature_count":
            (
                len(feature_names)
                if feature_names is not None
                else 0
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "interface":
            NETWORK_INTERFACE,

        "window_seconds":
            WINDOW_SECONDS,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import uvicorn

    # Required for Windows multiprocessing.
    mp.freeze_support()

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )