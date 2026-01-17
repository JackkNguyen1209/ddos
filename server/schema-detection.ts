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
  normalized: number;
  category: string;
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

const LABEL_MAPPINGS: Record<string, { value: number; category: string }> = {
  'benign': { value: 0, category: 'normal' },
  'normal': { value: 0, category: 'normal' },
  'legitimate': { value: 0, category: 'normal' },
  'safe': { value: 0, category: 'normal' },
  'ddos': { value: 1, category: 'ddos' },
  'drdos': { value: 1, category: 'ddos' },
  'drdos_dns': { value: 1, category: 'ddos_amplification' },
  'drdos_ldap': { value: 1, category: 'ddos_amplification' },
  'drdos_mssql': { value: 1, category: 'ddos_amplification' },
  'drdos_netbios': { value: 1, category: 'ddos_amplification' },
  'drdos_ntp': { value: 1, category: 'ddos_amplification' },
  'drdos_snmp': { value: 1, category: 'ddos_amplification' },
  'drdos_ssdp': { value: 1, category: 'ddos_amplification' },
  'drdos_udp': { value: 1, category: 'ddos_volumetric' },
  'drdos_udplag': { value: 1, category: 'ddos_volumetric' },
  'syn': { value: 1, category: 'ddos_protocol' },
  'syn_flood': { value: 1, category: 'ddos_protocol' },
  'udp': { value: 1, category: 'ddos_volumetric' },
  'udp_flood': { value: 1, category: 'ddos_volumetric' },
  'tftp': { value: 1, category: 'ddos_amplification' },
  'portmap': { value: 1, category: 'ddos_amplification' },
  'portscan': { value: 1, category: 'reconnaissance' },
  'netscan': { value: 1, category: 'reconnaissance' },
  'bruteforce': { value: 1, category: 'bruteforce' },
  'ssh_bruteforce': { value: 1, category: 'bruteforce' },
  'ftp_bruteforce': { value: 1, category: 'bruteforce' },
  'dos_slowloris': { value: 1, category: 'application_layer' },
  'dos_slowhttptest': { value: 1, category: 'application_layer' },
  'dos_hulk': { value: 1, category: 'application_layer' },
  'dos_goldeneye': { value: 1, category: 'application_layer' },
  'webddos': { value: 1, category: 'application_layer' },
  'bot': { value: 1, category: 'botnet' },
  'botnet': { value: 1, category: 'botnet' },
  'heartbleed': { value: 1, category: 'exploit' },
  'infiltration': { value: 1, category: 'infiltration' },
};

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
  
  if (LABEL_MAPPINGS[normalized]) {
    return {
      original: label,
      normalized: LABEL_MAPPINGS[normalized].value,
      category: LABEL_MAPPINGS[normalized].category,
    };
  }
  
  for (const [key, mapping] of Object.entries(LABEL_MAPPINGS)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return {
        original: label,
        normalized: mapping.value,
        category: mapping.category,
      };
    }
  }
  
  if (normalized.startsWith('drdos') || normalized.startsWith('ddos') || 
      normalized.includes('attack') || normalized.includes('malicious')) {
    return {
      original: label,
      normalized: 1,
      category: 'unknown_attack',
    };
  }
  
  return {
    original: label,
    normalized: 1,
    category: 'unknown',
  };
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
