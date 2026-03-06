import scapy.all as scapy
from scapy.layers.inet import IP
import numpy as np
import pandas as pd
import os

class PcapParser:
    def __init__(self, target_ip=None):
        self.target_ip = target_ip

    def parse(self, pcap_path):
        if not os.path.exists(pcap_path):
            print(f"File missing: {pcap_path}")
            return []

        try:
            packets = scapy.rdpcap(pcap_path)
        except Exception as e:
            print(f"Scapy can't read {pcap_path}: {e}")
            return []

        data = []
        if len(packets) == 0:
            return []

        start_ts = float(packets[0].time)

        for p in packets:
            # We only care about IP packets
            if IP not in p:
                continue

            ip_layer = p[IP]
            src = ip_layer.src
            dst = ip_layer.dst

            # Filter by target IP if set
            if self.target_ip and (self.target_ip not in [src, dst]):
                continue

            # Determine direction: 1 = Outgoing (Upload), -1 = Incoming (Download)
            direction = 0
            if self.target_ip:
                direction = 1 if src == self.target_ip else -1
            
            # Packet size (includes headers)
            size = len(p)
            ts = float(p.time)

            data.append({
                'ts': ts,
                'rel_time': ts - start_ts,
                'size': size,
                'dir': direction,
                'src': src,
                'dst': dst
            })

        return data

    def extract_stats(self, packet_list):
        # Convert to DF for easier stats
        if not packet_list:
            return {}

        df = pd.DataFrame(packet_list)
        
        # Split traffic
        outgoing = df[df['dir'] == 1]
        incoming = df[df['dir'] == -1]

        features = {}

        # 1. Packet Counts
        features['total_pkts'] = len(df)
        features['out_pkts'] = len(outgoing)
        features['in_pkts'] = len(incoming)

        # 2. Size Stats 
        # Helper lambda to keep code short
        calc_stats = lambda s, name: {
            f'{name}_mean': s.mean(),
            f'{name}_std': s.std(), 
            f'{name}_max': s.max(),
            f'{name}_min': s.min(),
            f'{name}_sum': s.sum()
        } if not s.empty else {k:0 for k in [f'{name}_mean', f'{name}_std', f'{name}_max', f'{name}_min', f'{name}_sum']}

        features.update(calc_stats(df['size'], 'all_bytes'))
        features.update(calc_stats(outgoing['size'], 'out_bytes'))
        features.update(calc_stats(incoming['size'], 'in_bytes'))

        # 3. Timing / IAT (Inter-arrival Time)
        df.sort_values('ts', inplace=True)
        iat = df['ts'].diff().dropna()
        features.update(calc_stats(iat, 'iat'))

        # IAT by direction
        if len(outgoing) > 1:
            out_iat = outgoing['ts'].diff().dropna()
            features.update(calc_stats(out_iat, 'out_iat'))
        else:
            features.update(calc_stats(pd.Series(dtype=float), 'out_iat'))

        if len(incoming) > 1:
            in_iat = incoming['ts'].diff().dropna()
            features.update(calc_stats(in_iat, 'in_iat'))
        else:
            features.update(calc_stats(pd.Series(dtype=float), 'in_iat'))

        # 4. Bursts

        if not df.empty:
            df['burst_id'] = (df['dir'] != df['dir'].shift()).cumsum()
            
            # aggregations
            bursts = df.groupby('burst_id')
            b_sizes = bursts['size'].sum()
            b_durations = bursts['ts'].max() - bursts['ts'].min()
            b_counts = bursts['size'].count()

            features.update(calc_stats(b_sizes, 'burst_size'))
            features.update(calc_stats(b_durations, 'burst_time'))
            features.update(calc_stats(b_counts, 'burst_len'))
            features['num_bursts'] = len(bursts)
        else:
             # fill zeros
            features.update(calc_stats(pd.Series(dtype=float), 'burst_size'))
            features.update(calc_stats(pd.Series(dtype=float), 'burst_time'))
            features.update(calc_stats(pd.Series(dtype=float), 'burst_len'))
            features['num_bursts'] = 0

        # Flow duration
        features['flow_duration'] = df['rel_time'].max() if not df.empty else 0

        return features

if __name__ == "__main__":
    # quick test
    pp = PcapParser()
    print("PcapParser loaded.")
