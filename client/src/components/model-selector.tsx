import { useState } from "react";
import { Brain, Check, Play, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { ML_MODELS, type MLModelType } from "@shared/schema";

interface ModelSelectorProps {
  selectedModels: MLModelType[];
  onModelsChange: (models: MLModelType[]) => void;
  onAnalyze: () => void;
  isAnalyzing?: boolean;
  disabled?: boolean;
}

export function ModelSelector({
  selectedModels,
  onModelsChange,
  onAnalyze,
  isAnalyzing,
  disabled,
}: ModelSelectorProps) {
  const toggleModel = (model: MLModelType) => {
    if (selectedModels.includes(model)) {
      onModelsChange(selectedModels.filter((m) => m !== model));
    } else {
      onModelsChange([...selectedModels, model]);
    }
  };

  const selectAll = () => {
    if (selectedModels.length === ML_MODELS.length) {
      onModelsChange([]);
    } else {
      onModelsChange(ML_MODELS.map((m) => m.type));
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Chọn Mô Hình ML
            </CardTitle>
            <CardDescription>
              Chọn một hoặc nhiều thuật toán để phân tích và so sánh kết quả
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={selectAll}
            disabled={disabled}
            data-testid="button-select-all-models"
          >
            {selectedModels.length === ML_MODELS.length ? "Bỏ chọn tất cả" : "Chọn tất cả"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {ML_MODELS.map((model) => {
            const isSelected = selectedModels.includes(model.type);
            return (
              <div
                key={model.type}
                className={cn(
                  "relative rounded-lg border p-4 transition-all cursor-pointer hover-elevate",
                  isSelected
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                )}
                onClick={() => !disabled && toggleModel(model.type)}
                data-testid={`card-model-${model.type}`}
              >
                <div className="flex items-start gap-3">
                  <Checkbox
                    id={model.type}
                    checked={isSelected}
                    onCheckedChange={() => toggleModel(model.type)}
                    disabled={disabled}
                    className="mt-0.5"
                    data-testid={`checkbox-model-${model.type}`}
                  />
                  <div className="flex-1 min-w-0">
                    <Label
                      htmlFor={model.type}
                      className="text-sm font-medium cursor-pointer"
                    >
                      {model.name}
                    </Label>
                    <p className="text-xs text-muted-foreground mt-1">
                      {model.description}
                    </p>
                  </div>
                  {isSelected && (
                    <div className="absolute right-2 top-2">
                      <div className="rounded-full bg-primary p-1">
                        <Check className="h-3 w-3 text-primary-foreground" />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-4 border-t">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Info className="h-4 w-4" />
            <span>Đã chọn {selectedModels.length} mô hình</span>
          </div>
          <Button
            onClick={onAnalyze}
            disabled={disabled || selectedModels.length === 0 || isAnalyzing}
            className="sm:ml-auto"
            data-testid="button-start-analysis"
          >
            {isAnalyzing ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Đang phân tích...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Bắt đầu phân tích
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
