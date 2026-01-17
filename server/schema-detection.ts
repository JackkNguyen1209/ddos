/**
 * Schema Detection and Column Normalization Service
 * Detects CICFlowMeter vs Event/Log formats and normalizes column names
 */

export type SchemaType = 'cicflowmeter' | 'event_log' | 'unknown';

export interface FeatureReport {
  schemaType: SchemaType;
  schemaConfidence: number;
  foundFeatures: string[];
  missingFeatures: string[];
  foundPercentage: number;
  nanCount: number;
  infCount: number;
  totalValues: number;
  nanPercentage: number;
  infPercentage: number;
  isReliable: boolean;
  warnings: string[];
  columnMappings: Record<string, string>;
  normalizedColumns: string[];
}

export interface LabelMapping {
  original: string;
  normalized: number;          // 0 = normal, 1 = anomaly/attack
  category: string;            // Detailed category
  severity: 'low' | 'medium' | 'high' | 'critical';
  isAttack: boolean;           // true = attack, false = anomaly only
  description: string;
}

// Label categories with severity and attack classification
export type LabelCategory = 
  | 'normal'           // Bình thường
  | 'ddos'             // DDoS chung
  | 'ddos_volumetric'  // DDoS lưu lượng lớn (UDP flood, ICMP flood)
  | 'ddos_protocol'    // DDoS khai thác giao thức (SYN flood, TCP)
  | 'ddos_amplification' // DDoS khuếch đại (DNS, NTP, SSDP)
  | 'ddos_application' // DDoS lớp ứng dụng (HTTP flood, Slowloris)
  | 'reconnaissance'   // Quét thăm dò (Port scan, Network scan)
  | 'bruteforce'       // Tấn công brute force
  | 'exploit'          // Khai thác lỗ hổng
  | 'malware'          // Mã độc, botnet
  | 'infiltration'     // Xâm nhập
  | 'anomaly_traffic'  // Bất thường lưu lượng (không rõ attack)
  | 'anomaly_behavior' // Bất thường hành vi
  | 'anomaly_protocol' // Bất thường giao thức
  | 'anomaly_resource' // Bất thường tài nguyên
  | 'custom';          // Tùy chỉnh

export interface LabelConfig {
  value: number;
  category: LabelCategory;
  severity: 'low' | 'medium' | 'high' | 'critical';
  isAttack: boolean;
  description: string;
}

const COLUMN_ALIASES: Record<string, string[]> = {
  'flow_packets_s': ['pps', 'packets_per_second', 'pkts_s', 'packet_rate', 'flow_pkts_s'],
  'flow_bytes_s': ['byte_rate', 'bytes_per_second', 'bps', 'flow_bps', 'byterate'],
  'flow_duration': ['duration', 'flow_dur', 'dur', 'connection_duration'],
  'total_fwd_packets': ['fwd_packets', 'fwd_pkts', 'tot_fwd_pkts', 'src_pkts'],
  'total_bwd_packets': ['bwd_packets', 'bwd_pkts', 'tot_bwd_pkts', 'dst_pkts'],
  'total_length_of_fwd_packets': ['fwd_bytes', 'fwd_length', 'src_bytes'],
  'total_length_of_bwd_packets': ['bwd_bytes', 'bwd_length', 'dst_bytes'],
  'fwd_packet_length_max': ['fwd_pkt_len_max', 'max_fwd_pkt_len'],
  'fwd_packet_length_min': ['fwd_pkt_len_min', 'min_fwd_pkt_len'],
  'fwd_packet_length_mean': ['fwd_pkt_len_mean', 'avg_fwd_pkt_len'],
  'bwd_packet_length_max': ['bwd_pkt_len_max', 'max_bwd_pkt_len'],
  'bwd_packet_length_min': ['bwd_pkt_len_min', 'min_bwd_pkt_len'],
  'bwd_packet_length_mean': ['bwd_pkt_len_mean', 'avg_bwd_pkt_len'],
  'flow_iat_mean': ['iat_mean', 'mean_iat', 'avg_iat'],
  'flow_iat_std': ['iat_std', 'std_iat'],
  'flow_iat_max': ['iat_max', 'max_iat'],
  'flow_iat_min': ['iat_min', 'min_iat'],
  'fwd_iat_total': ['fwd_iat_sum', 'total_fwd_iat'],
  'fwd_iat_mean': ['avg_fwd_iat', 'mean_fwd_iat'],
  'bwd_iat_total': ['bwd_iat_sum', 'total_bwd_iat'],
  'bwd_iat_mean': ['avg_bwd_iat', 'mean_bwd_iat'],
  'fwd_psh_flags': ['psh_flag_count', 'fwd_psh'],
  'fwd_urg_flags': ['urg_flag_count', 'fwd_urg'],
  'fwd_header_length': ['fwd_hdr_len', 'src_header_len'],
  'bwd_header_length': ['bwd_hdr_len', 'dst_header_len'],
  'min_packet_length': ['min_pkt_len', 'pkt_len_min'],
  'max_packet_length': ['max_pkt_len', 'pkt_len_max'],
  'packet_length_mean': ['avg_pkt_len', 'pkt_len_mean', 'mean_pkt_len'],
  'packet_length_std': ['pkt_len_std', 'std_pkt_len'],
  'packet_length_variance': ['pkt_len_var', 'var_pkt_len'],
  'fin_flag_count': ['fin_flags', 'tcp_fin'],
  'syn_flag_count': ['syn_flags', 'tcp_syn'],
  'rst_flag_count': ['rst_flags', 'tcp_rst'],
  'psh_flag_count': ['psh_flags', 'tcp_psh'],
  'ack_flag_count': ['ack_flags', 'tcp_ack'],
  'urg_flag_count': ['urg_flags', 'tcp_urg'],
  'cwe_flag_count': ['cwe_flags', 'tcp_cwe'],
  'ece_flag_count': ['ece_flags', 'tcp_ece'],
  'down_up_ratio': ['ratio', 'download_upload_ratio'],
  'average_packet_size': ['avg_pkt_size', 'mean_pkt_size'],
  'avg_fwd_segment_size': ['fwd_seg_size_avg', 'avg_fwd_seg'],
  'avg_bwd_segment_size': ['bwd_seg_size_avg', 'avg_bwd_seg'],
  'init_win_bytes_forward': ['init_fwd_win', 'fwd_init_win'],
  'init_win_bytes_backward': ['init_bwd_win', 'bwd_init_win'],
  'active_mean': ['avg_active', 'mean_active_time'],
  'active_std': ['std_active', 'active_time_std'],
  'idle_mean': ['avg_idle', 'mean_idle_time'],
  'idle_std': ['std_idle', 'idle_time_std'],
  'label': ['class', 'attack_type', 'category', 'target', 'classification'],
  'src_ip': ['source_ip', 'srcip', 'source_address', 'src_addr'],
  'dst_ip': ['dest_ip', 'dstip', 'destination_ip', 'dest_addr', 'dst_addr'],
  'src_port': ['source_port', 'srcport', 'sport'],
  'dst_port': ['dest_port', 'dstport', 'dport', 'destination_port'],
  'protocol': ['proto', 'ip_protocol', 'transport_protocol'],
  'timestamp': ['time', 'datetime', 'ts', 'date', 'event_time'],
};

const CICFLOWMETER_REQUIRED_FEATURES = [
  'flow_duration',
  'total_fwd_packets',
  'total_bwd_packets',
  'flow_bytes_s',
  'flow_packets_s',
  'flow_iat_mean',
  'fwd_packet_length_mean',
  'bwd_packet_length_mean',
];

const CICFLOWMETER_OPTIONAL_FEATURES = [
  'fwd_iat_mean',
  'bwd_iat_mean',
  'fin_flag_count',
  'syn_flag_count',
  'rst_flag_count',
  'psh_flag_count',
  'ack_flag_count',
  'urg_flag_count',
  'average_packet_size',
  'active_mean',
  'idle_mean',
  'init_win_bytes_forward',
  'init_win_bytes_backward',
  'min_packet_length',
  'max_packet_length',
  'packet_length_mean',
  'packet_length_std',
];

const EVENT_LOG_FEATURES = [
  'timestamp',
  'src_ip',
  'dst_ip',
  'src_port',
  'dst_port',
  'protocol',
  'event_type',
  'severity',
  'message',
  'action',
  'status',
  'user',
  'bytes',
  'packets',
];

// Comprehensive label mappings supporting multiple datasets
const LABEL_MAPPINGS: Record<string, LabelConfig> = {
  // === NORMAL / BENIGN ===
  'benign': { value: 0, category: 'normal', severity: 'low', isAttack: false, description: 'Lưu lượng bình thường' },
  'normal': { value: 0, category: 'normal', severity: 'low', isAttack: false, description: 'Lưu lượng bình thường' },
  'legitimate': { value: 0, category: 'normal', severity: 'low', isAttack: false, description: 'Lưu lượng hợp lệ' },
  'safe': { value: 0, category: 'normal', severity: 'low', isAttack: false, description: 'Lưu lượng an toàn' },
  'clean': { value: 0, category: 'normal', severity: 'low', isAttack: false, description: 'Lưu lượng sạch' },
  '0': { value: 0, category: 'normal', severity: 'low', isAttack: false, description: 'Label 0 - Bình thường' },
  
  // === DDoS GENERAL ===
  'ddos': { value: 1, category: 'ddos', severity: 'critical', isAttack: true, description: 'Tấn công DDoS chung' },
  'drdos': { value: 1, category: 'ddos', severity: 'critical', isAttack: true, description: 'Tấn công DRDoS' },
  'dos': { value: 1, category: 'ddos', severity: 'high', isAttack: true, description: 'Tấn công DoS' },
  'attack': { value: 1, category: 'ddos', severity: 'high', isAttack: true, description: 'Tấn công chung' },
  '1': { value: 1, category: 'ddos', severity: 'high', isAttack: true, description: 'Label 1 - Tấn công' },
  
  // === DDoS AMPLIFICATION (Khuếch đại) ===
  'drdos_dns': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại DNS' },
  'drdos_ldap': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại LDAP' },
  'drdos_mssql': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại MSSQL' },
  'drdos_netbios': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại NetBIOS' },
  'drdos_ntp': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại NTP' },
  'drdos_snmp': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại SNMP' },
  'drdos_ssdp': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại SSDP' },
  'drdos_chargen': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại Chargen' },
  'drdos_memcached': { value: 1, category: 'ddos_amplification', severity: 'critical', isAttack: true, description: 'DRDoS khuếch đại Memcached' },
  'tftp': { value: 1, category: 'ddos_amplification', severity: 'high', isAttack: true, description: 'Tấn công TFTP amplification' },
  'portmap': { value: 1, category: 'ddos_amplification', severity: 'high', isAttack: true, description: 'Tấn công Portmap amplification' },
  
  // === DDoS VOLUMETRIC (Lưu lượng lớn) ===
  'drdos_udp': { value: 1, category: 'ddos_volumetric', severity: 'critical', isAttack: true, description: 'DRDoS UDP flood' },
  'drdos_udplag': { value: 1, category: 'ddos_volumetric', severity: 'critical', isAttack: true, description: 'DRDoS UDP lag' },
  'udp': { value: 1, category: 'ddos_volumetric', severity: 'high', isAttack: true, description: 'UDP flood' },
  'udp_flood': { value: 1, category: 'ddos_volumetric', severity: 'critical', isAttack: true, description: 'UDP flood attack' },
  'icmp_flood': { value: 1, category: 'ddos_volumetric', severity: 'high', isAttack: true, description: 'ICMP flood attack' },
  'ping_flood': { value: 1, category: 'ddos_volumetric', severity: 'high', isAttack: true, description: 'Ping flood attack' },
  'smurf': { value: 1, category: 'ddos_volumetric', severity: 'high', isAttack: true, description: 'Smurf attack' },
  
  // === DDoS PROTOCOL (Khai thác giao thức) ===
  'syn': { value: 1, category: 'ddos_protocol', severity: 'critical', isAttack: true, description: 'SYN attack' },
  'syn_flood': { value: 1, category: 'ddos_protocol', severity: 'critical', isAttack: true, description: 'SYN flood attack' },
  'tcp_flood': { value: 1, category: 'ddos_protocol', severity: 'high', isAttack: true, description: 'TCP flood attack' },
  'ack_flood': { value: 1, category: 'ddos_protocol', severity: 'high', isAttack: true, description: 'ACK flood attack' },
  'rst_flood': { value: 1, category: 'ddos_protocol', severity: 'high', isAttack: true, description: 'RST flood attack' },
  'fin_flood': { value: 1, category: 'ddos_protocol', severity: 'high', isAttack: true, description: 'FIN flood attack' },
  'land': { value: 1, category: 'ddos_protocol', severity: 'medium', isAttack: true, description: 'LAND attack' },
  'teardrop': { value: 1, category: 'ddos_protocol', severity: 'medium', isAttack: true, description: 'Teardrop attack' },
  
  // === DDoS APPLICATION LAYER (Lớp ứng dụng) ===
  'dos_slowloris': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'Slowloris attack' },
  'dos_slowhttptest': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'Slow HTTP test attack' },
  'dos_hulk': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'HULK attack' },
  'dos_goldeneye': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'GoldenEye attack' },
  'webddos': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'Web DDoS attack' },
  'http_flood': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'HTTP flood attack' },
  'rudy': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'RUDY attack' },
  'apache_killer': { value: 1, category: 'ddos_application', severity: 'critical', isAttack: true, description: 'Apache Killer attack' },
  
  // === RECONNAISSANCE (Quét thăm dò) ===
  'portscan': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Port scanning' },
  'netscan': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Network scanning' },
  'reconnaissance': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Quét thăm dò chung' },
  'probe': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Probing attack' },
  'nmap': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Nmap scanning' },
  'satan': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'SATAN scan' },
  'ipsweep': { value: 1, category: 'reconnaissance', severity: 'low', isAttack: true, description: 'IP sweep scan' },
  'mscan': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Mscan attack' },
  'saint': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Saint scan' },
  
  // === BRUTEFORCE (Tấn công dò mật khẩu) ===
  'bruteforce': { value: 1, category: 'bruteforce', severity: 'high', isAttack: true, description: 'Brute force attack' },
  'ssh_bruteforce': { value: 1, category: 'bruteforce', severity: 'high', isAttack: true, description: 'SSH brute force' },
  'ftp_bruteforce': { value: 1, category: 'bruteforce', severity: 'high', isAttack: true, description: 'FTP brute force' },
  'guess_passwd': { value: 1, category: 'bruteforce', severity: 'high', isAttack: true, description: 'Password guessing' },
  'snmpguess': { value: 1, category: 'bruteforce', severity: 'medium', isAttack: true, description: 'SNMP guess attack' },
  'httptunnel': { value: 1, category: 'bruteforce', severity: 'medium', isAttack: true, description: 'HTTP tunnel' },
  
  // === EXPLOIT (Khai thác lỗ hổng) ===
  'heartbleed': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'Heartbleed exploit' },
  'shellshock': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'Shellshock exploit' },
  'sql_injection': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'SQL injection' },
  'xss': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Cross-site scripting' },
  'buffer_overflow': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'Buffer overflow' },
  'rootkit': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'Rootkit attack' },
  'loadmodule': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Load module attack' },
  'perl': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Perl attack' },
  'phf': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'PHF attack' },
  'r2l': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Remote to Local attack' },
  'u2r': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'User to Root attack' },
  
  // === MALWARE / BOTNET ===
  'bot': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Bot/Botnet traffic' },
  'botnet': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Botnet attack' },
  'worm': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Worm infection' },
  'trojan': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Trojan activity' },
  'backdoor': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Backdoor activity' },
  'virus': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Virus activity' },
  'ransomware': { value: 1, category: 'malware', severity: 'critical', isAttack: true, description: 'Ransomware attack' },
  'spyware': { value: 1, category: 'malware', severity: 'high', isAttack: true, description: 'Spyware activity' },
  
  // === INFILTRATION (Xâm nhập) ===
  'infiltration': { value: 1, category: 'infiltration', severity: 'critical', isAttack: true, description: 'Xâm nhập hệ thống' },
  'apt': { value: 1, category: 'infiltration', severity: 'critical', isAttack: true, description: 'Advanced Persistent Threat' },
  'exfiltration': { value: 1, category: 'infiltration', severity: 'critical', isAttack: true, description: 'Data exfiltration' },
  
  // === ANOMALY - TRAFFIC (Bất thường lưu lượng - KHÔNG RÕ LÀ TẤN CÔNG) ===
  'anomaly': { value: 1, category: 'anomaly_traffic', severity: 'medium', isAttack: false, description: 'Bất thường lưu lượng chung' },
  'unusual_traffic': { value: 1, category: 'anomaly_traffic', severity: 'medium', isAttack: false, description: 'Lưu lượng bất thường' },
  'high_bandwidth': { value: 1, category: 'anomaly_traffic', severity: 'low', isAttack: false, description: 'Băng thông cao bất thường' },
  'spike': { value: 1, category: 'anomaly_traffic', severity: 'medium', isAttack: false, description: 'Đột biến lưu lượng' },
  'burst': { value: 1, category: 'anomaly_traffic', severity: 'low', isAttack: false, description: 'Burst traffic' },
  'congestion': { value: 1, category: 'anomaly_traffic', severity: 'low', isAttack: false, description: 'Tắc nghẽn mạng' },
  
  // === ANOMALY - BEHAVIOR (Bất thường hành vi) ===
  'suspicious': { value: 1, category: 'anomaly_behavior', severity: 'medium', isAttack: false, description: 'Hành vi đáng ngờ' },
  'unusual_connection': { value: 1, category: 'anomaly_behavior', severity: 'medium', isAttack: false, description: 'Kết nối bất thường' },
  'abnormal_session': { value: 1, category: 'anomaly_behavior', severity: 'medium', isAttack: false, description: 'Phiên bất thường' },
  'policy_violation': { value: 1, category: 'anomaly_behavior', severity: 'low', isAttack: false, description: 'Vi phạm chính sách' },
  'unusual_login': { value: 1, category: 'anomaly_behavior', severity: 'medium', isAttack: false, description: 'Đăng nhập bất thường' },
  'failed_login': { value: 1, category: 'anomaly_behavior', severity: 'low', isAttack: false, description: 'Đăng nhập thất bại' },
  
  // === ANOMALY - PROTOCOL (Bất thường giao thức) ===
  'malformed_packet': { value: 1, category: 'anomaly_protocol', severity: 'medium', isAttack: false, description: 'Gói tin sai định dạng' },
  'protocol_violation': { value: 1, category: 'anomaly_protocol', severity: 'medium', isAttack: false, description: 'Vi phạm giao thức' },
  'fragmentation': { value: 1, category: 'anomaly_protocol', severity: 'low', isAttack: false, description: 'Phân mảnh bất thường' },
  'invalid_header': { value: 1, category: 'anomaly_protocol', severity: 'medium', isAttack: false, description: 'Header không hợp lệ' },
  
  // === ANOMALY - RESOURCE (Bất thường tài nguyên) ===
  'high_cpu': { value: 1, category: 'anomaly_resource', severity: 'medium', isAttack: false, description: 'CPU cao bất thường' },
  'high_memory': { value: 1, category: 'anomaly_resource', severity: 'medium', isAttack: false, description: 'Memory cao bất thường' },
  'disk_full': { value: 1, category: 'anomaly_resource', severity: 'high', isAttack: false, description: 'Ổ đĩa đầy' },
  'resource_exhaustion': { value: 1, category: 'anomaly_resource', severity: 'high', isAttack: false, description: 'Cạn kiệt tài nguyên' },
  
  // === UNSW-NB15 Dataset Labels ===
  'analysis': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Analysis attack (UNSW-NB15)' },
  'fuzzers': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Fuzzing attack (UNSW-NB15)' },
  'generic': { value: 1, category: 'ddos', severity: 'high', isAttack: true, description: 'Generic attack (UNSW-NB15)' },
  'exploits': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'Exploits (UNSW-NB15)' },
  'shellcode': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'Shellcode attack (UNSW-NB15)' },
  
  // === NSL-KDD Dataset Labels ===
  'neptune': { value: 1, category: 'ddos_protocol', severity: 'high', isAttack: true, description: 'Neptune attack (NSL-KDD)' },
  'pod': { value: 1, category: 'ddos_volumetric', severity: 'high', isAttack: true, description: 'Ping of Death (NSL-KDD)' },
  'apache2': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'Apache2 attack (NSL-KDD)' },
  'back': { value: 1, category: 'ddos_application', severity: 'high', isAttack: true, description: 'Back attack (NSL-KDD)' },
  'mailbomb': { value: 1, category: 'ddos_volumetric', severity: 'high', isAttack: true, description: 'Mailbomb attack (NSL-KDD)' },
  'processtable': { value: 1, category: 'ddos', severity: 'high', isAttack: true, description: 'Process table attack (NSL-KDD)' },
  'udpstorm': { value: 1, category: 'ddos_volumetric', severity: 'critical', isAttack: true, description: 'UDP Storm (NSL-KDD)' },
  'warezmaster': { value: 1, category: 'infiltration', severity: 'high', isAttack: true, description: 'Warezmaster (NSL-KDD)' },
  'snmpgetattack': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'SNMP Get attack (NSL-KDD)' },
  'multihop': { value: 1, category: 'infiltration', severity: 'high', isAttack: true, description: 'Multihop attack (NSL-KDD)' },
  'ftp_write': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'FTP Write attack (NSL-KDD)' },
  'imap': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'IMAP attack (NSL-KDD)' },
  'spy': { value: 1, category: 'infiltration', severity: 'high', isAttack: true, description: 'Spy attack (NSL-KDD)' },
  'xlock': { value: 1, category: 'exploit', severity: 'medium', isAttack: true, description: 'Xlock attack (NSL-KDD)' },
  'xsnoop': { value: 1, category: 'reconnaissance', severity: 'medium', isAttack: true, description: 'Xsnoop attack (NSL-KDD)' },
  'sendmail': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Sendmail attack (NSL-KDD)' },
  'named': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Named attack (NSL-KDD)' },
  'xterm': { value: 1, category: 'exploit', severity: 'high', isAttack: true, description: 'Xterm attack (NSL-KDD)' },
  'ps': { value: 1, category: 'reconnaissance', severity: 'low', isAttack: true, description: 'PS attack (NSL-KDD)' },
  'sqlattack': { value: 1, category: 'exploit', severity: 'critical', isAttack: true, description: 'SQL Attack (NSL-KDD)' },
};

// Custom label definitions storage (user-defined)
let customLabelMappings: Record<string, LabelConfig> = {};

export function normalizeColumnName(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/[\s\-\.]+/g, '_')
    .replace(/[^a-z0-9_]/g, '')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
}

export function findColumnMapping(columns: string[]): Record<string, string> {
  const mappings: Record<string, string> = {};
  const normalizedColumns = columns.map(c => normalizeColumnName(c));
  
  for (const [canonical, aliases] of Object.entries(COLUMN_ALIASES)) {
    const allNames = [canonical, ...aliases];
    
    for (let i = 0; i < columns.length; i++) {
      const normalized = normalizedColumns[i];
      if (allNames.includes(normalized)) {
        mappings[columns[i]] = canonical;
        break;
      }
    }
  }
  
  for (let i = 0; i < columns.length; i++) {
    if (!Object.keys(mappings).includes(columns[i])) {
      mappings[columns[i]] = normalizedColumns[i];
    }
  }
  
  return mappings;
}

export function detectSchemaType(columns: string[]): { type: SchemaType; confidence: number } {
  const mappings = findColumnMapping(columns);
  const normalizedCols = Object.values(mappings);
  
  let cicflowmeterScore = 0;
  let cicflowmeterTotal = CICFLOWMETER_REQUIRED_FEATURES.length + CICFLOWMETER_OPTIONAL_FEATURES.length;
  
  for (const feature of CICFLOWMETER_REQUIRED_FEATURES) {
    if (normalizedCols.includes(feature)) {
      cicflowmeterScore += 2;
    }
  }
  cicflowmeterTotal += CICFLOWMETER_REQUIRED_FEATURES.length;
  
  for (const feature of CICFLOWMETER_OPTIONAL_FEATURES) {
    if (normalizedCols.includes(feature)) {
      cicflowmeterScore += 1;
    }
  }
  
  let eventLogScore = 0;
  for (const feature of EVENT_LOG_FEATURES) {
    if (normalizedCols.includes(feature)) {
      eventLogScore += 1;
    }
  }
  
  const cicflowmeterConfidence = (cicflowmeterScore / cicflowmeterTotal) * 100;
  const eventLogConfidence = (eventLogScore / EVENT_LOG_FEATURES.length) * 100;
  
  if (cicflowmeterConfidence >= 40) {
    return { type: 'cicflowmeter', confidence: Math.min(cicflowmeterConfidence, 100) };
  } else if (eventLogConfidence >= 40) {
    return { type: 'event_log', confidence: Math.min(eventLogConfidence, 100) };
  }
  
  return { type: 'unknown', confidence: Math.max(cicflowmeterConfidence, eventLogConfidence) };
}

export function mapLabel(label: string): LabelMapping {
  const normalized = label
    .toLowerCase()
    .trim()
    .replace(/[\s\-\.]+/g, '_')
    .replace(/[^a-z0-9_]/g, '');
  
  // Check custom mappings first (user-defined)
  if (customLabelMappings[normalized]) {
    const mapping = customLabelMappings[normalized];
    return {
      original: label,
      normalized: mapping.value,
      category: mapping.category,
      severity: mapping.severity,
      isAttack: mapping.isAttack,
      description: mapping.description,
    };
  }
  
  // Check built-in mappings
  if (LABEL_MAPPINGS[normalized]) {
    const mapping = LABEL_MAPPINGS[normalized];
    return {
      original: label,
      normalized: mapping.value,
      category: mapping.category,
      severity: mapping.severity,
      isAttack: mapping.isAttack,
      description: mapping.description,
    };
  }
  
  // Partial match in built-in mappings
  for (const [key, mapping] of Object.entries(LABEL_MAPPINGS)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return {
        original: label,
        normalized: mapping.value,
        category: mapping.category,
        severity: mapping.severity,
        isAttack: mapping.isAttack,
        description: mapping.description,
      };
    }
  }
  
  // Heuristic detection for unknown attack patterns
  if (normalized.startsWith('drdos') || normalized.startsWith('ddos') || 
      normalized.includes('attack') || normalized.includes('malicious') ||
      normalized.includes('flood') || normalized.includes('exploit')) {
    return {
      original: label,
      normalized: 1,
      category: 'custom',
      severity: 'medium',
      isAttack: true,
      description: `Phát hiện tấn công: ${label}`,
    };
  }
  
  // Heuristic detection for anomaly patterns (not confirmed attack)
  if (normalized.includes('anomaly') || normalized.includes('unusual') ||
      normalized.includes('suspicious') || normalized.includes('abnormal')) {
    return {
      original: label,
      normalized: 1,
      category: 'anomaly_behavior',
      severity: 'medium',
      isAttack: false,
      description: `Bất thường phát hiện: ${label}`,
    };
  }
  
  // Default: treat as unknown anomaly (not confirmed attack)
  return {
    original: label,
    normalized: 1,
    category: 'custom',
    severity: 'low',
    isAttack: false,
    description: `Label không xác định: ${label}`,
  };
}

// Functions for custom label management
export function addCustomLabel(
  labelName: string, 
  config: { 
    isAttack: boolean; 
    category?: LabelCategory; 
    severity?: 'low' | 'medium' | 'high' | 'critical';
    description?: string;
  }
): void {
  const normalized = labelName.toLowerCase().trim().replace(/[\s\-\.]+/g, '_');
  customLabelMappings[normalized] = {
    value: config.isAttack ? 1 : 0,
    category: config.category || (config.isAttack ? 'custom' : 'normal'),
    severity: config.severity || 'medium',
    isAttack: config.isAttack,
    description: config.description || `Custom label: ${labelName}`,
  };
}

export function removeCustomLabel(labelName: string): boolean {
  const normalized = labelName.toLowerCase().trim().replace(/[\s\-\.]+/g, '_');
  if (customLabelMappings[normalized]) {
    delete customLabelMappings[normalized];
    return true;
  }
  return false;
}

export function getCustomLabels(): Record<string, LabelConfig> {
  return { ...customLabelMappings };
}

export function getAllLabelMappings(): Record<string, LabelConfig> {
  return { ...LABEL_MAPPINGS, ...customLabelMappings };
}

export function setCustomLabels(labels: Record<string, LabelConfig>): void {
  customLabelMappings = { ...labels };
}

export function analyzeFeatureUsage(
  columns: string[],
  data: any[][],
  schemaType: SchemaType
): FeatureReport {
  const mappings = findColumnMapping(columns);
  const normalizedCols = Object.values(mappings);
  
  const requiredFeatures = schemaType === 'cicflowmeter' 
    ? CICFLOWMETER_REQUIRED_FEATURES 
    : EVENT_LOG_FEATURES.slice(0, 6);
    
  const allFeatures = schemaType === 'cicflowmeter'
    ? [...CICFLOWMETER_REQUIRED_FEATURES, ...CICFLOWMETER_OPTIONAL_FEATURES]
    : EVENT_LOG_FEATURES;
  
  const foundFeatures: string[] = [];
  const missingFeatures: string[] = [];
  
  for (const feature of allFeatures) {
    if (normalizedCols.includes(feature)) {
      foundFeatures.push(feature);
    } else {
      missingFeatures.push(feature);
    }
  }
  
  let nanCount = 0;
  let infCount = 0;
  let totalValues = 0;
  
  for (const row of data) {
    for (const value of row) {
      totalValues++;
      if (value === null || value === undefined || value === '' || 
          (typeof value === 'number' && isNaN(value)) ||
          (typeof value === 'string' && (value.toLowerCase() === 'nan' || value.toLowerCase() === 'null'))) {
        nanCount++;
      }
      if (typeof value === 'number' && !isFinite(value)) {
        infCount++;
      }
      if (typeof value === 'string' && (value.toLowerCase() === 'inf' || value.toLowerCase() === '-inf' || value.toLowerCase() === 'infinity')) {
        infCount++;
      }
    }
  }
  
  const foundPercentage = (foundFeatures.length / allFeatures.length) * 100;
  const nanPercentage = totalValues > 0 ? (nanCount / totalValues) * 100 : 0;
  const infPercentage = totalValues > 0 ? (infCount / totalValues) * 100 : 0;
  
  const warnings: string[] = [];
  
  if (foundPercentage < 60) {
    warnings.push(`Chỉ tìm thấy ${foundPercentage.toFixed(1)}% features cần thiết. Kết quả có thể không đáng tin cậy.`);
  }
  
  if (nanPercentage > 10) {
    warnings.push(`Tỉ lệ giá trị NaN/null cao (${nanPercentage.toFixed(1)}%). Dữ liệu cần được làm sạch.`);
  }
  
  if (infPercentage > 1) {
    warnings.push(`Phát hiện giá trị Infinity (${infPercentage.toFixed(1)}%). Có thể ảnh hưởng đến kết quả phân tích.`);
  }
  
  const requiredMissing = requiredFeatures.filter(f => !normalizedCols.includes(f));
  if (requiredMissing.length > 0) {
    warnings.push(`Thiếu ${requiredMissing.length} features bắt buộc: ${requiredMissing.slice(0, 3).join(', ')}${requiredMissing.length > 3 ? '...' : ''}`);
  }
  
  const { type, confidence } = detectSchemaType(columns);
  
  return {
    schemaType: type,
    schemaConfidence: confidence,
    foundFeatures,
    missingFeatures,
    foundPercentage,
    nanCount,
    infCount,
    totalValues,
    nanPercentage,
    infPercentage,
    isReliable: foundPercentage >= 60 && nanPercentage <= 20 && infPercentage <= 5,
    warnings,
    columnMappings: mappings,
    normalizedColumns: normalizedCols,
  };
}

export function normalizeDataset(
  columns: string[],
  data: any[][],
  mappings: Record<string, string>
): { columns: string[]; data: any[][] } {
  const normalizedColumns = columns.map(c => mappings[c] || normalizeColumnName(c));
  
  const labelIndex = normalizedColumns.findIndex(c => c === 'label');
  
  const normalizedData = data.map(row => {
    return row.map((value, i) => {
      if (i === labelIndex && typeof value === 'string') {
        const labelMapping = mapLabel(value);
        return labelMapping.normalized;
      }
      
      if (typeof value === 'string') {
        const num = parseFloat(value);
        if (!isNaN(num) && isFinite(num)) {
          return num;
        }
        if (value.toLowerCase() === 'nan' || value.toLowerCase() === 'null' || value === '') {
          return 0;
        }
        if (value.toLowerCase() === 'inf' || value.toLowerCase() === 'infinity') {
          return Number.MAX_SAFE_INTEGER;
        }
        if (value.toLowerCase() === '-inf') {
          return Number.MIN_SAFE_INTEGER;
        }
      }
      
      if (typeof value === 'number') {
        if (isNaN(value)) return 0;
        if (!isFinite(value)) return value > 0 ? Number.MAX_SAFE_INTEGER : Number.MIN_SAFE_INTEGER;
      }
      
      return value;
    });
  });
  
  return { columns: normalizedColumns, data: normalizedData };
}

export function getModelForSchema(schemaType: SchemaType): string[] {
  switch (schemaType) {
    case 'cicflowmeter':
      return ['random_forest', 'decision_tree', 'knn', 'naive_bayes', 'logistic_regression', 'lucid'];
    case 'event_log':
      return ['isolation_forest', 'lof', 'naive_bayes', 'logistic_regression'];
    default:
      return ['isolation_forest', 'lof', 'random_forest'];
  }
}

export function getLabelStats(data: any[][], labelIndex: number): Record<string, { count: number; percentage: number; category: string }> {
  const stats: Record<string, { count: number; category: string }> = {};
  
  for (const row of data) {
    const label = String(row[labelIndex] || 'unknown');
    const mapping = mapLabel(label);
    
    if (!stats[label]) {
      stats[label] = { count: 0, category: mapping.category };
    }
    stats[label].count++;
  }
  
  const total = data.length;
  const result: Record<string, { count: number; percentage: number; category: string }> = {};
  
  for (const [label, info] of Object.entries(stats)) {
    result[label] = {
      count: info.count,
      percentage: (info.count / total) * 100,
      category: info.category,
    };
  }
  
  return result;
}
