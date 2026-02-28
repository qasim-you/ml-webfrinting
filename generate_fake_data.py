import scapy.all as scapy
from scapy.layers.inet import IP, UDP, Ether
import os
import random
import time
import shutil

# Configuration
DATA_DIR = "data"
CLASSES = {
    "turn_on_lights": {
        "duration_mean": 2.0, "duration_std": 0.5,
        "packets_mean": 50, "packets_std": 10,
        "burst_pattern": "short_response" 
    },
    "play_music": {
        "duration_mean": 5.0, "duration_std": 1.0,
        "packets_mean": 300, "packets_std": 50,
        "burst_pattern": "streaming"
    },
    "whats_the_weather": {
        "duration_mean": 3.0, "duration_std": 0.5,
        "packets_mean": 120, "packets_std": 20,
        "burst_pattern": "medium_response"
    }
}
SAMPLES_PER_CLASS = 20
TARGET_IP = "192.168.1.100"
GATEWAY_IP = "192.168.1.1"

def generate_trace(label, filename):
    params = CLASSES[label]
    
    # Randomize generic params
    duration = max(0.5, random.gauss(params["duration_mean"], params["duration_std"]))
    num_packets = int(max(10, random.gauss(params["packets_mean"], params["packets_std"])))
    
    packets = []
    start_time = time.time()
    current_time = start_time
    
    # Traffic Model:
    # 1. User Command (Outgoing to Cloud) - Short burst
    # 2. Processing pause
    # 3. Server Response (Incoming from Cloud) - Pattern depends on class
    
    # 1. Command (Outgoing)
    for _ in range(random.randint(5, 15)):
        pkt_size = random.randint(100, 400)
        pkt = Ether() / IP(src=TARGET_IP, dst=GATEWAY_IP) / UDP(sport=12345, dport=443) / ("X" * pkt_size)
        pkt.time = current_time
        packets.append(pkt)
        current_time += random.uniform(0.001, 0.01)
        
    # Processing delay
    current_time += random.uniform(0.1, 0.5)
    
    # 2. Response (Incoming)
    remaining_packets = num_packets - len(packets)
    
    if params["burst_pattern"] == "streaming":
        # High throughput, large packets
        for _ in range(remaining_packets):
            pkt_size = random.randint(800, 1400)
            pkt = Ether() / IP(src=GATEWAY_IP, dst=TARGET_IP) / UDP(sport=443, dport=12345) / ("Y" * pkt_size)
            pkt.time = current_time
            packets.append(pkt)
            current_time += random.uniform(0.0005, 0.002) # Fast inter-arrival
            
    elif params["burst_pattern"] == "short_response":
        # Concise TTS response
        for _ in range(remaining_packets):
            pkt_size = random.randint(200, 600)
            pkt = Ether() / IP(src=GATEWAY_IP, dst=TARGET_IP) / UDP(sport=443, dport=12345) / ("Y" * pkt_size)
            pkt.time = current_time
            packets.append(pkt)
            current_time += random.uniform(0.002, 0.01)

    else: # medium
        for _ in range(remaining_packets):
            pkt_size = random.randint(400, 1000)
            pkt = Ether() / IP(src=GATEWAY_IP, dst=TARGET_IP) / UDP(sport=443, dport=12345) / ("Y" * pkt_size)
            pkt.time = current_time
            packets.append(pkt)
            current_time += random.uniform(0.001, 0.005)

    scapy.wrpcap(filename, packets)

def main():
    print(f"Generating synthetic PCAP data in '{DATA_DIR}'...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for label in CLASSES.keys():
        class_dir = os.path.join(DATA_DIR, label)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"  Generating {SAMPLES_PER_CLASS} traces for '{label}'...")
        for i in range(SAMPLES_PER_CLASS):
            fname = os.path.join(class_dir, f"trace_{i:03d}.pcap")
            generate_trace(label, fname)
            
    print("Done. Data generation complete.")

if __name__ == "__main__":
    main()
