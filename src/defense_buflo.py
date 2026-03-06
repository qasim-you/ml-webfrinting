import numpy as np
import random

class BuFLOShim:
    def __init__(self, interval=0.02, size=1400, min_len=2.0):
        self.interval = interval   
        self.size = size          
        self.min_len = min_len    

    def apply(self, packets):
        """
        Simulate BuFLO defense.
        1. Packets are queued.
        2. Every 'interval', send a packet of 'size'.
        3. If no real packet, send dummy.
        """
        if not packets:
            return []

        # sort by time first
        packets.sort(key=lambda x: x['ts'])
        
        start = packets[0]['ts']
        end = packets[-1]['ts']
        dur = max(end - start, self.min_len)

        defended = []
        curr_time = start
        

        q = packets[:] # copy
        
        steps = int(np.ceil(dur / self.interval))
        
        for i in range(steps + 5): 
            now = start + (i * self.interval)
            
            # Find next valid packet
            has_packet = False
            if len(q) > 0 and q[0]['ts'] <= now:
                # Take it
                real_p = q.pop(0)
                direction = real_p['dir']
                has_packet = True
            
            if has_packet:
               
                p = {
                    'ts': now,
                    'size': self.size,
                    'dir': direction,
                    'rel_time': now - start,
                    'is_dummy': False
                }
            else:
                # Send dummy
    
                p = {
                    'ts': now,
                    'size': self.size,
                    'dir': 1, 
                    'rel_time': now - start,
                    'is_dummy': True
                }
            
            defended.append(p)

        return defended

    def get_overhead(self, orig, def_pkts):
        orig_bytes = sum(p['size'] for p in orig)
        def_bytes = sum(p['size'] for p in def_pkts)
        
        if orig_bytes == 0: return 0
        return (def_bytes - orig_bytes) / orig_bytes
