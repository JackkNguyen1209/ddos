import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { CheckCircle, XCircle, Tag, MessageSquare, Plus, Trash2, RefreshCw, Layers } from "lucide-react";

interface UserFeedback {
  id: number;
  rowIndex: number;
  originalLabel: string | null;
  correctedLabel: string;
  isAttack: boolean;
  category: string;
  severity: string;
  userNotes: string | null;
  isApplied: boolean;
  createdAt: string;
}

interface UserTag {
  id: number;
  tagName: string;
  tagColor: string;
  description: string | null;
  isAttackTag: boolean;
  usageCount: number;
}

interface ReviewSummary {
  totalRows: number;
  feedbackCount: number;
  pendingFeedback: number;
  appliedFeedback: number;
  availableTags: number;
  datasetName: string;
}

const CATEGORIES = [
  { value: 'normal', label: 'Normal' },
  { value: 'ddos', label: 'DDoS' },
  { value: 'ddos_volumetric', label: 'DDoS Volumetric' },
  { value: 'ddos_protocol', label: 'DDoS Protocol' },
  { value: 'ddos_amplification', label: 'DDoS Amplification' },
  { value: 'ddos_application', label: 'DDoS Application' },
  { value: 'reconnaissance', label: 'Reconnaissance' },
  { value: 'bruteforce', label: 'Bruteforce' },
  { value: 'exploit', label: 'Exploit' },
  { value: 'malware', label: 'Malware' },
  { value: 'infiltration', label: 'Infiltration' },
  { value: 'anomaly_traffic', label: 'Anomaly Traffic' },
  { value: 'anomaly_behavior', label: 'Anomaly Behavior' },
  { value: 'custom', label: 'Custom' },
];

const SEVERITIES = [
  { value: 'low', label: 'Thấp', color: 'bg-blue-100 text-blue-800' },
  { value: 'medium', label: 'Trung bình', color: 'bg-yellow-100 text-yellow-800' },
  { value: 'high', label: 'Cao', color: 'bg-orange-100 text-orange-800' },
  { value: 'critical', label: 'Nghiêm trọng', color: 'bg-red-100 text-red-800' },
];

const TAG_COLORS = [
  '#ef4444', '#f97316', '#f59e0b', '#84cc16', '#22c55e', 
  '#14b8a6', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899'
];

export function ReviewPanel() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("feedback");
  const [feedbackDialogOpen, setFeedbackDialogOpen] = useState(false);
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  
  // Form states
  const [newFeedback, setNewFeedback] = useState({
    rowIndex: 0,
    originalLabel: '',
    correctedLabel: '',
    isAttack: false,
    category: 'normal',
    severity: 'medium',
    userNotes: '',
  });
  
  const [newTag, setNewTag] = useState({
    tagName: '',
    tagColor: '#3b82f6',
    description: '',
    isAttackTag: false,
  });

  // Queries
  const { data: summary } = useQuery<ReviewSummary>({
    queryKey: ['/api/review/summary'],
  });

  const { data: feedbackData } = useQuery<{ feedback: UserFeedback[] }>({
    queryKey: ['/api/feedback'],
  });

  const { data: tagsData } = useQuery<{ tags: UserTag[] }>({
    queryKey: ['/api/tags'],
  });

  // Mutations
  const addFeedbackMutation = useMutation({
    mutationFn: async (data: typeof newFeedback) => {
      const res = await apiRequest('POST', '/api/feedback', data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/feedback'] });
      queryClient.invalidateQueries({ queryKey: ['/api/review/summary'] });
      setFeedbackDialogOpen(false);
      setNewFeedback({ rowIndex: 0, originalLabel: '', correctedLabel: '', isAttack: false, category: 'normal', severity: 'medium', userNotes: '' });
      toast({ title: "Đã lưu phản hồi", description: "Phản hồi của bạn đã được ghi nhận" });
    },
    onError: () => {
      toast({ title: "Lỗi", description: "Không thể lưu phản hồi", variant: "destructive" });
    }
  });

  const applyFeedbackMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest('POST', '/api/feedback/apply');
      return res.json();
    },
    onSuccess: (data: { applied: number; message: string }) => {
      queryClient.invalidateQueries({ queryKey: ['/api/feedback'] });
      queryClient.invalidateQueries({ queryKey: ['/api/review/summary'] });
      queryClient.invalidateQueries({ queryKey: ['/api/labels'] });
      toast({ title: "Đã áp dụng", description: data.message });
    },
    onError: () => {
      toast({ title: "Lỗi", description: "Không thể áp dụng phản hồi", variant: "destructive" });
    }
  });

  const deleteFeedbackMutation = useMutation({
    mutationFn: async (id: number) => {
      const res = await apiRequest('DELETE', `/api/feedback/${id}`);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/feedback'] });
      queryClient.invalidateQueries({ queryKey: ['/api/review/summary'] });
      toast({ title: "Đã xóa", description: "Phản hồi đã được xóa" });
    }
  });

  const addTagMutation = useMutation({
    mutationFn: async (data: typeof newTag) => {
      const res = await apiRequest('POST', '/api/tags', data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/tags'] });
      queryClient.invalidateQueries({ queryKey: ['/api/review/summary'] });
      setTagDialogOpen(false);
      setNewTag({ tagName: '', tagColor: '#3b82f6', description: '', isAttackTag: false });
      toast({ title: "Đã tạo tag", description: "Tag mới đã được tạo thành công" });
    },
    onError: () => {
      toast({ title: "Lỗi", description: "Không thể tạo tag", variant: "destructive" });
    }
  });

  const deleteTagMutation = useMutation({
    mutationFn: async (id: number) => {
      const res = await apiRequest('DELETE', `/api/tags/${id}`);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/tags'] });
      queryClient.invalidateQueries({ queryKey: ['/api/review/summary'] });
      toast({ title: "Đã xóa", description: "Tag đã được xóa" });
    }
  });

  const feedback = feedbackData?.feedback || [];
  const tags = tagsData?.tags || [];

  return (
    <Card data-testid="card-review-panel">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4" />
          Đánh giá & Gắn tag
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {summary && (
          <div className="grid grid-cols-2 gap-2 text-xs" data-testid="container-review-summary">
            <div className="p-2 bg-muted rounded">
              <div className="text-muted-foreground">Phản hồi chờ xử lý</div>
              <div className="text-lg font-bold" data-testid="text-pending-count">{summary.pendingFeedback}</div>
            </div>
            <div className="p-2 bg-muted rounded">
              <div className="text-muted-foreground">Tags đã tạo</div>
              <div className="text-lg font-bold" data-testid="text-tags-count">{summary.availableTags}</div>
            </div>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="feedback" data-testid="tab-feedback">
              <MessageSquare className="h-3 w-3 mr-1" />
              Phản hồi ({feedback.length})
            </TabsTrigger>
            <TabsTrigger value="tags" data-testid="tab-tags">
              <Tag className="h-3 w-3 mr-1" />
              Tags ({tags.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="feedback" className="space-y-3">
            <div className="flex gap-2">
              <Dialog open={feedbackDialogOpen} onOpenChange={setFeedbackDialogOpen}>
                <DialogTrigger asChild>
                  <Button size="sm" variant="outline" data-testid="button-add-feedback">
                    <Plus className="h-3 w-3 mr-1" />
                    Thêm phản hồi
                  </Button>
                </DialogTrigger>
                <DialogContent data-testid="dialog-add-feedback">
                  <DialogHeader>
                    <DialogTitle>Thêm phản hồi đánh giá</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <Label>Dòng số</Label>
                        <Input 
                          type="number" 
                          value={newFeedback.rowIndex}
                          onChange={(e) => setNewFeedback(prev => ({ ...prev, rowIndex: parseInt(e.target.value) || 0 }))}
                          data-testid="input-row-index"
                        />
                      </div>
                      <div>
                        <Label>Nhãn gốc</Label>
                        <Input 
                          value={newFeedback.originalLabel}
                          onChange={(e) => setNewFeedback(prev => ({ ...prev, originalLabel: e.target.value }))}
                          placeholder="Nhãn hiện tại"
                          data-testid="input-original-label"
                        />
                      </div>
                    </div>
                    <div>
                      <Label>Nhãn đúng</Label>
                      <Input 
                        value={newFeedback.correctedLabel}
                        onChange={(e) => setNewFeedback(prev => ({ ...prev, correctedLabel: e.target.value }))}
                        placeholder="Nhập nhãn chính xác"
                        data-testid="input-corrected-label"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <Label>Loại</Label>
                        <Select 
                          value={newFeedback.category} 
                          onValueChange={(v) => setNewFeedback(prev => ({ ...prev, category: v }))}
                        >
                          <SelectTrigger data-testid="select-category">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {CATEGORIES.map(c => (
                              <SelectItem key={c.value} value={c.value}>{c.label}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>Mức độ</Label>
                        <Select 
                          value={newFeedback.severity} 
                          onValueChange={(v) => setNewFeedback(prev => ({ ...prev, severity: v }))}
                        >
                          <SelectTrigger data-testid="select-severity">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {SEVERITIES.map(s => (
                              <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={newFeedback.isAttack}
                        onChange={(e) => setNewFeedback(prev => ({ ...prev, isAttack: e.target.checked }))}
                        data-testid="checkbox-is-attack"
                      />
                      <Label>Đây là tấn công</Label>
                    </div>
                    <div>
                      <Label>Ghi chú</Label>
                      <Textarea 
                        value={newFeedback.userNotes}
                        onChange={(e) => setNewFeedback(prev => ({ ...prev, userNotes: e.target.value }))}
                        placeholder="Ghi chú thêm về phản hồi này..."
                        data-testid="textarea-notes"
                      />
                    </div>
                    <Button 
                      className="w-full" 
                      onClick={() => addFeedbackMutation.mutate(newFeedback)}
                      disabled={!newFeedback.correctedLabel || addFeedbackMutation.isPending}
                      data-testid="button-submit-feedback"
                    >
                      {addFeedbackMutation.isPending ? 'Đang lưu...' : 'Lưu phản hồi'}
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>

              {summary && summary.pendingFeedback > 0 && (
                <Button 
                  size="sm" 
                  onClick={() => applyFeedbackMutation.mutate()}
                  disabled={applyFeedbackMutation.isPending}
                  data-testid="button-apply-feedback"
                >
                  <RefreshCw className={`h-3 w-3 mr-1 ${applyFeedbackMutation.isPending ? 'animate-spin' : ''}`} />
                  Áp dụng ({summary.pendingFeedback})
                </Button>
              )}
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto" data-testid="container-feedback-list">
              {feedback.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  Chưa có phản hồi nào. Thêm phản hồi để cải thiện độ chính xác.
                </p>
              ) : (
                feedback.map((fb, index) => (
                  <div 
                    key={fb.id} 
                    className="p-2 border rounded text-xs flex items-center justify-between gap-2"
                    data-testid={`feedback-item-${index}`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <Badge variant={fb.isAttack ? 'destructive' : 'secondary'} className="text-xs">
                          {fb.isAttack ? 'Attack' : 'Normal'}
                        </Badge>
                        <span className={`text-xs px-1.5 py-0.5 rounded ${SEVERITIES.find(s => s.value === fb.severity)?.color || ''}`}>
                          {fb.severity}
                        </span>
                        {fb.isApplied ? (
                          <CheckCircle className="h-3 w-3 text-green-500" />
                        ) : (
                          <XCircle className="h-3 w-3 text-orange-500" />
                        )}
                      </div>
                      <div className="text-muted-foreground truncate">
                        Dòng {fb.rowIndex}: {fb.originalLabel || '?'} → {fb.correctedLabel}
                      </div>
                    </div>
                    <Button 
                      size="icon" 
                      variant="ghost" 
                      className="h-6 w-6"
                      onClick={() => deleteFeedbackMutation.mutate(fb.id)}
                      data-testid={`button-delete-feedback-${index}`}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="tags" className="space-y-3">
            <Dialog open={tagDialogOpen} onOpenChange={setTagDialogOpen}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline" data-testid="button-add-tag">
                  <Plus className="h-3 w-3 mr-1" />
                  Tạo tag mới
                </Button>
              </DialogTrigger>
              <DialogContent data-testid="dialog-add-tag">
                <DialogHeader>
                  <DialogTitle>Tạo tag mới</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label>Tên tag</Label>
                    <Input 
                      value={newTag.tagName}
                      onChange={(e) => setNewTag(prev => ({ ...prev, tagName: e.target.value }))}
                      placeholder="Ví dụ: suspicious_traffic"
                      data-testid="input-tag-name"
                    />
                  </div>
                  <div>
                    <Label>Màu</Label>
                    <div className="flex gap-2 flex-wrap mt-1">
                      {TAG_COLORS.map(color => (
                        <button
                          key={color}
                          className={`w-6 h-6 rounded-full border-2 ${newTag.tagColor === color ? 'border-foreground' : 'border-transparent'}`}
                          style={{ backgroundColor: color }}
                          onClick={() => setNewTag(prev => ({ ...prev, tagColor: color }))}
                          data-testid={`button-color-${color.replace('#', '')}`}
                        />
                      ))}
                    </div>
                  </div>
                  <div>
                    <Label>Mô tả</Label>
                    <Input 
                      value={newTag.description}
                      onChange={(e) => setNewTag(prev => ({ ...prev, description: e.target.value }))}
                      placeholder="Mô tả ngắn về tag này"
                      data-testid="input-tag-description"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <input 
                      type="checkbox" 
                      checked={newTag.isAttackTag}
                      onChange={(e) => setNewTag(prev => ({ ...prev, isAttackTag: e.target.checked }))}
                      data-testid="checkbox-is-attack-tag"
                    />
                    <Label>Đánh dấu là tag tấn công</Label>
                  </div>
                  <Button 
                    className="w-full" 
                    onClick={() => addTagMutation.mutate(newTag)}
                    disabled={!newTag.tagName || addTagMutation.isPending}
                    data-testid="button-submit-tag"
                  >
                    {addTagMutation.isPending ? 'Đang tạo...' : 'Tạo tag'}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <div className="space-y-2 max-h-48 overflow-y-auto" data-testid="container-tags-list">
              {tags.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  Chưa có tag nào. Tạo tag để phân loại dữ liệu.
                </p>
              ) : (
                tags.map((tag, index) => (
                  <div 
                    key={tag.id} 
                    className="p-2 border rounded text-xs flex items-center justify-between gap-2"
                    data-testid={`tag-item-${index}`}
                  >
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-4 h-4 rounded-full" 
                        style={{ backgroundColor: tag.tagColor }}
                      />
                      <div>
                        <div className="font-medium flex items-center gap-1">
                          {tag.tagName}
                          {tag.isAttackTag && (
                            <Badge variant="destructive" className="text-[10px] px-1 py-0">Attack</Badge>
                          )}
                        </div>
                        {tag.description && (
                          <div className="text-muted-foreground">{tag.description}</div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">x{tag.usageCount}</span>
                      <Button 
                        size="icon" 
                        variant="ghost" 
                        className="h-6 w-6"
                        onClick={() => deleteTagMutation.mutate(tag.id)}
                        data-testid={`button-delete-tag-${index}`}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
